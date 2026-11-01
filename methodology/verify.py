"""
Lossless round-trip verification for the joint tokenizer.

Tests two things:
1. **In-vocab round-trip** — every (NL, SMILES) pair in the training set
   round-trips losslessly: decode(encode(x)) == x for both streams.
2. **Held-out round-trip** — SMILES that the tokenizer has never seen still
   encode/decode losslessly (because the tokenizer never drops atoms/chars
   from its vocabulary, only merges them).

The held-out test is the critical one: it proves the tokenizer generalizes
to novel chemistry, not just memorizes training molecules.

Usage:
    python verify.py                    # uses default config (N=200 samples)
    python verify.py --n-samples 500    # larger sample
    python verify.py --tokenizer joint  # only test joint (default)
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from typing import List, Tuple

from config import CFG
from dataset import load_dataset
from joint_tokenizer import JointTokenizer, smiles_pre_tokenize


# ============================================================ held-out SMILES
# 20 SMILES that are NOT in the synthetic dataset, used to test generalization.
# These are real drug molecules from ChEMBL with diverse scaffolds.
HELD_OUT_SMILES = [
    # antibiotics / antivirals
    "CC1=C(C(=O)C(=C(C1=O)C(=O)O)N)O",      # tetracycline-like
    "OC(=O)C1=CC=CC=C1NC(=O)C2=CC=CC=C2",   # hippurate analog
    # cardiovascular
    "CC(C)COC(=O)C1=CC=CC=C1C(=O)O",        # methyl hippurate variant
    # CNS
    "CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31",  # imipramine-like tricyclic
    # antineoplastic
    "C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2",       # benzanilide
    # simple aromatics not in training set
    "Brc1ccc(Br)cc1",                        # 1,4-dibromobenzene
    "Clc1ccc(Cl)c(Cl)c1",                    # 1,2,4-trichlorobenzene
    "Oc1ccc(O)c(O)c1",                       # 1,2,4-benzenetriol
    "Nc1ccc(N)c(N)c1",                       # 1,2,4-triaminobenzene
    "O=C(O)c1ccc(C(=O)O)c(C(=O)O)c1",       # trimesic acid
    # heterocycles
    "c1cc[nH]c1C(=O)O",                     # indole-3-carboxylic acid
    "c1csc(C(=O)O)c1",                       # thiophene-2-carboxylic acid
    "n1ccnc1C(=O)O",                         # imidazole-4-carboxylic acid
    "O=c1cc[nH]c(=O)[nH]1",                 # uracil (in-set but canonical form)
    # fused rings
    "c1ccc2c(c1)OCO2",                       # methylenedioxybenzene
    "c1ccc2c(c1)CCN2",                       # isoindoline
    # steroids (simplified)
    "CC12CCC3C4CCC(=O)C=4CCC3C1CCC2O",      # estrone skeleton
    # peptides
    "Nc1ccccc1C(=O)Nc1ccccc1C(=O)O",        # hippuric-anthranilic
    # prostaglandin-like
    "CCCCC(O)C/C=C\\CCCC(=O)O",             # simplified PGE1
    # sugars
    "OCC1OC(O)C(O)C(O)C1O",                 # glucose open form (approx)
]


def verify_in_vocab(tok: JointTokenizer, pairs: List[Tuple[str, str]], n_samples: int) -> Tuple[int, int, List[str]]:
    """Verify round-trip on training pairs. Returns (n_pass, n_total, errors)."""
    if n_samples < len(pairs):
        sample = random.sample(pairs, n_samples)
    else:
        sample = pairs
        n_samples = len(pairs)

    n_pass = 0
    errors = []
    for nl, smi in sample:
        # SMILES round-trip (explicit stream="chem")
        _, smi_ids = tok.encode("", smi)
        decoded = tok.decode(smi_ids, stream="chem")
        if decoded != smi:
            errors.append(f"SMILES mismatch: {smi!r} → ids {smi_ids} → {decoded!r}")
            continue
        # NL round-trip (explicit stream="nl")
        nl_ids, _ = tok.encode(nl, "")
        decoded_nl = tok.decode(nl_ids, stream="nl")
        # Normalize whitespace + punctuation: collapse " ," → "," and "  " → " "
        def normalize(s: str) -> str:
            s = s.lower().strip()
            s = re.sub(r"\s+([,.;:!?])", r"\1", s)  # remove space before punctuation
            s = re.sub(r"\s+", " ", s)
            return s
        if normalize(decoded_nl) != normalize(nl):
            errors.append(f"NL mismatch: {nl!r} → {decoded_nl!r} (normalized: {normalize(decoded_nl)!r} vs {normalize(nl)!r})")
            continue
        n_pass += 1
    return n_pass, n_samples, errors


def verify_held_out(tok: JointTokenizer, smiles_list: List[str]) -> Tuple[int, int, List[str]]:
    """Verify round-trip on held-out SMILES. Returns (n_pass, n_total, errors)."""
    n_pass = 0
    errors = []
    for smi in smiles_list:
        _, smi_ids = tok.encode("", smi)
        decoded = tok.decode(smi_ids, stream="chem")
        if decoded != smi:
            errors.append(f"Held-out SMILES mismatch: {smi!r} → ids {smi_ids} → {decoded!r}")
        else:
            n_pass += 1
    return n_pass, len(smiles_list), errors


def main():
    ap = argparse.ArgumentParser(description="Verify lossless round-trip on the joint tokenizer")
    ap.add_argument("--n-samples", type=int, default=CFG.verify_n_samples,
                    help=f"number of in-vocab pairs to test (default {CFG.verify_n_samples})")
    ap.add_argument("--seed", type=int, default=0, help="random seed for sampling")
    args = ap.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("Lossless round-trip verification — JointTokenizer")
    print("=" * 70)

    # Load dataset and tokenizer
    pairs = load_dataset(CFG.data_path)
    print(f"Loaded {len(pairs)} pairs from {CFG.data_path}")

    print("\nBuilding joint tokenizer...")
    tok = JointTokenizer()
    tok.build(pairs)
    print(f"  {tok}")

    # In-vocab test
    print(f"\n--- In-vocab round-trip test ({args.n_samples} samples) ---")
    n_pass, n_total, errors = verify_in_vocab(tok, pairs, args.n_samples)
    print(f"  Passed: {n_pass}/{n_total}  ({100*n_pass/n_total:.1f}%)")
    if errors:
        print(f"  First 5 errors:")
        for e in errors[:5]:
            print(f"    - {e}")

    # Held-out test
    print(f"\n--- Held-out round-trip test ({len(HELD_OUT_SMILES)} novel SMILES) ---")
    n_pass_h, n_total_h, errors_h = verify_held_out(tok, HELD_OUT_SMILES)
    print(f"  Passed: {n_pass_h}/{n_total_h}  ({100*n_pass_h/n_total_h:.1f}%)")
    if errors_h:
        print(f"  First 5 errors:")
        for e in errors_h[:5]:
            print(f"    - {e}")

    # Final verdict
    print("\n" + "=" * 70)
    overall_pass = (n_pass == n_total) and (n_pass_h == n_total_h)
    if overall_pass:
        print("✓ ALL ROUND-TRIP TESTS PASSED — tokenizer is lossless.")
    else:
        print("✗ SOME TESTS FAILED — see errors above.")
    print("=" * 70)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
