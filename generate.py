"""
End-to-end inference: NL prompt -> SMILES -> RDKit validation -> properties -> 2D image.

Implements the post-processing pipeline described in Section 3.5 of the article:
  - Generate SMILES from an NL prompt via the lightweight decoder
  - Validate via RDKit (parse + canonicalize)
  - Repair trivial parenthesis-balance errors
  - Extract the longest valid SMILES prefix (handles noisy model output)
  - Compute physico-chemical descriptors: MW, LogP, TPSA, QED, num_rings
  - Compute Tanimoto similarity to a reference molecule (optional)
  - Render a 2D structure image (PNG)

Usage:
    # Train a model first
    python train.py

    # Then generate
    python generate.py "generate a painkiller with a benzene ring"
    python generate.py "synthesize aspirin" --render
    python generate.py "create caffeine" --render --reference "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

Author: Methodology component #3 - inference + post-processing pipeline.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

# Silence RDKit's verbose error logging before any imports
import logging
logging.getLogger("rdkit").setLevel(logging.CRITICAL)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdKit.*")
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

import torch

from joint_tokenizer import JointTokenizer
from model import DecoderConfig, LightweightDecoder, quantize_int8


@dataclass
class MoleculeResult:
    """Result of a single text-to-molecule inference call."""
    prompt: str
    raw_smiles: str
    canonical_smiles: str
    valid: bool
    repair_attempted: bool
    molecular_weight: float
    logp: float
    tpsa: float
    qed: float
    num_rings: int
    num_rotatable_bonds: int
    latency_ms: float
    num_tokens_generated: int
    tanimoto_to_reference: Optional[float] = None

    def as_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        if not self.valid:
            return f"[INVALID] prompt={self.prompt!r}  raw={self.raw_smiles!r}  ({self.latency_ms:.1f} ms)"
        return (f"[VALID]   prompt={self.prompt!r}  "
                f"SMILES={self.canonical_smiles}  "
                f"MW={self.molecular_weight:.1f}  LogP={self.logp:.2f}  "
                f"TPSA={self.tpsa:.1f}  QED={self.qed:.3f}  "
                f"({self.latency_ms:.1f} ms)")


class MoleculeGenerator:
    """End-to-end NL -> molecule -> properties pipeline."""

    def __init__(
        self,
        tokenizer: JointTokenizer,
        model: LightweightDecoder,
        device: str = "cpu",
        use_int8: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        if use_int8:
            self.model = quantize_int8(self.model).to(device)
        self.model.eval()
        self.cfg: DecoderConfig = model.cfg

    # ------------------------------------------------------------- load
    @classmethod
    def from_checkpoint(cls, ckpt_dir: str = "ckpt", device: str = "cpu", use_int8: bool = False) -> "MoleculeGenerator":
        """Load a trained model + tokenizer from a checkpoint directory."""
        tok_path = os.path.join(ckpt_dir, "tokenizer.json")
        model_path = os.path.join(ckpt_dir, "model.pt")
        if not (os.path.exists(tok_path) and os.path.exists(model_path)):
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_dir}/. Run `python train.py` first."
            )
        tok = JointTokenizer.load(tok_path)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        cfg = DecoderConfig(**ckpt["cfg"])
        model = LightweightDecoder(cfg)
        model.load_state_dict(ckpt["model_state"])
        return cls(tok, model, device=device, use_int8=use_int8)

    # ------------------------------------------------------------- repair
    @staticmethod
    def _repair_parentheses(s: str) -> str:
        """Append missing closing parens to balance the string."""
        depth = 0
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                return s
        if depth > 0:
            s = s + ")" * depth
        return s

    @staticmethod
    def _validate_smiles(smiles: str) -> Optional[str]:
        """Return canonical SMILES if valid, else None."""
        if not smiles:
            return None
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    @staticmethod
    def _extract_longest_valid_smiles(raw: str) -> Tuple[str, str]:
        """Given a raw generated string, find the longest valid SMILES prefix.

        Returns (canonical_smiles_or_empty, raw_smiles_used).
        The model often generates a valid molecule then keeps generating
        garbage; this extracts the valid portion.
        """
        candidates = [raw]
        if " " in raw:
            candidates.append(raw.split()[0])
        for end in range(len(raw), 0, -1):
            substr = raw[:end]
            if substr and substr[-1] in ")]0123456789" and len(substr) >= 4:
                candidates.append(substr)
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        for cand in unique_candidates:
            canonical = MoleculeGenerator._validate_smiles(cand)
            if canonical is not None:
                return canonical, cand
        return "", raw

    # ------------------------------------------------------------- properties
    @staticmethod
    def _compute_properties(canonical_smiles: str) -> Dict[str, float]:
        if not canonical_smiles:
            return {
                "molecular_weight": 0.0, "logp": 0.0, "tpsa": 0.0,
                "qed": 0.0, "num_rings": 0, "num_rotatable_bonds": 0,
            }
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED, rdMolDescriptors
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            return {
                "molecular_weight": 0.0, "logp": 0.0, "tpsa": 0.0,
                "qed": 0.0, "num_rings": 0, "num_rotatable_bonds": 0,
            }
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "qed": QED.qed(mol),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        }

    # ------------------------------------------------------------- similarity
    @staticmethod
    def tanimoto_similarity(smi_a: str, smi_b: str, radius: int = 2, n_bits: int = 2048) -> float:
        """Tanimoto similarity between two SMILES using Morgan fingerprints."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit import DataStructs
            ma = Chem.MolFromSmiles(smi_a)
            mb = Chem.MolFromSmiles(smi_b)
            if ma is None or mb is None:
                return 0.0
            fa = AllChem.GetMorganFingerprintAsBitVect(ma, radius, nBits=n_bits)
            fb = AllChem.GetMorganFingerprintAsBitVect(mb, radius, nBits=n_bits)
            return DataStructs.TanimotoSimilarity(fa, fb)
        except Exception:
            return 0.0

    # ------------------------------------------------------------- render
    @staticmethod
    def render_2d(smiles: str, out_path: str, size: tuple = (400, 400)) -> bool:
        """Render a 2D molecule structure PNG. Returns True on success."""
        if not smiles:
            return False
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            Draw.MolToFile(mol, out_path, size=size)
            return True
        except Exception as e:
            print(f"  (rendering failed: {e})")
            return False

    # ------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.85,
        top_p: float = 0.9,
        reference_smiles: Optional[str] = None,
    ) -> MoleculeResult:
        """Generate a molecule from a natural-language prompt."""
        t_start = time.perf_counter()

        nl_ids, _ = self.tokenizer.encode(prompt, smiles="")
        input_ids = [self.cfg.bos_id] + nl_ids + [self.cfg.nl_sep_id]
        prompt_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        out = self.model.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )
        generated_ids = out[0, len(input_ids):].tolist()
        if self.cfg.eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(self.cfg.eos_id)]

        raw_smiles = self.tokenizer.decode(generated_ids, stream="chem").strip()

        # Validate (with repair + longest-valid-prefix extraction)
        canonical, extracted_raw = self._extract_longest_valid_smiles(raw_smiles)
        repair_attempted = False
        if not canonical:
            repair_attempted = True
            repaired = self._repair_parentheses(raw_smiles)
            canonical, extracted_raw = self._extract_longest_valid_smiles(repaired)
            if canonical:
                raw_smiles = repaired
        else:
            raw_smiles = extracted_raw

        valid = bool(canonical)
        canonical = canonical or ""

        props = self._compute_properties(canonical)
        latency_ms = (time.perf_counter() - t_start) * 1000

        tanimoto = None
        if reference_smiles and valid:
            tanimoto = self.tanimoto_similarity(canonical, reference_smiles)

        return MoleculeResult(
            prompt=prompt,
            raw_smiles=raw_smiles,
            canonical_smiles=canonical,
            valid=valid,
            repair_attempted=repair_attempted,
            latency_ms=latency_ms,
            num_tokens_generated=len(generated_ids),
            tanimoto_to_reference=tanimoto,
            **props,
        )


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="Generate a molecule from a natural-language prompt")
    ap.add_argument("prompt", type=str, help='natural-language instruction, e.g. "generate a painkiller with a benzene ring"')
    ap.add_argument("--ckpt", type=str, default="ckpt", help="checkpoint directory (default: ckpt)")
    ap.add_argument("--max-tokens", type=int, default=80, help="max tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.85, help="sampling temperature (higher = more random)")
    ap.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling threshold")
    ap.add_argument("--render", action="store_true", help="render 2D structure PNG")
    ap.add_argument("--render-path", type=str, default="results/generated_molecule.png")
    ap.add_argument("--reference", type=str, default=None, help="reference SMILES for Tanimoto similarity")
    ap.add_argument("--no-int8", action="store_true", help="disable INT8 quantization")
    ap.add_argument("--json", action="store_true", help="output as JSON")
    args = ap.parse_args()

    try:
        gen = MoleculeGenerator.from_checkpoint(args.ckpt, use_int8=not args.no_int8)
        print(f"Loaded model from {args.ckpt}/  (vocab={len(gen.tokenizer)})\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run `python train.py` first to train a model.")
        return

    result = gen.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reference_smiles=args.reference,
    )

    if args.json:
        import json
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(result.summary())
        if result.valid:
            print(f"\n  Prompt              : {result.prompt}")
            print(f"  Generated SMILES    : {result.canonical_smiles}")
            print(f"  Molecular weight    : {result.molecular_weight:.2f} Da")
            print(f"  LogP                : {result.logp:.2f}")
            print(f"  TPSA                : {result.tpsa:.2f}")
            print(f"  QED (drug-likeness) : {result.qed:.3f}")
            print(f"  Rings               : {result.num_rings}")
            print(f"  Rotatable bonds     : {result.num_rotatable_bonds}")
            print(f"  Latency             : {result.latency_ms:.1f} ms")
            print(f"  Tokens generated    : {result.num_tokens_generated}")
            if result.tanimoto_to_reference is not None:
                print(f"  Tanimoto to ref     : {result.tanimoto_to_reference:.3f}")
        else:
            print(f"\n  Raw output          : {result.raw_smiles!r}")
            print(f"  Repair attempted    : {result.repair_attempted}")
            print(f"  Latency             : {result.latency_ms:.1f} ms")

        if args.render and result.valid:
            success = MoleculeGenerator.render_2d(result.canonical_smiles, args.render_path)
            if success:
                print(f"\n  2D structure saved  : {args.render_path}")


if __name__ == "__main__":
    main()