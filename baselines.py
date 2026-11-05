"""
Baseline tokenizers for the ablation table.

Implements two baselines to compare against the JointTokenizer:

1. **CharLevelTokenizer** — one token per SMILES character or NL character.
   No BPE merges. This is the simplest possible tokenizer and represents
   the "no compression" extreme.

2. **SeparateBPE** — two independent BPE tokenizers (NL + Chem), each with
   its own vocabulary. The two vocabularies do NOT share tokens. This
   represents the "separate-vocabulary" approach used by prior work like
   MolGPT and Text2Mol-base.

Both baselines implement the same interface as JointTokenizer so the
comparison script can treat them uniformly.

Author: Methodology module — joint tokenizer PoC.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from config import CFG
from joint_tokenizer import (
    BOS_TOKEN, EOS_TOKEN, NL_SEP, PAD_TOKEN, SPECIAL_TOKENS, UNK_TOKEN,
    SMILES_REGEX,
    _get_pair_counts, _merge_pair, nl_pre_tokenize, smiles_pre_tokenize,
)

# SMILES base alphabet — always included in vocab regardless of corpus
SMILES_BASE_ALPHABET = list("0123456789BCFHIKNOPSclnopsr().[]=#@+-/\\:")


# ============================================================ Char-level
class CharLevelTokenizer:
    """One token per character. NL and Chem share the same ASCII vocab."""

    def __init__(self, vocab_size: int = CFG.char_vocab_size):
        self.vocab_size = vocab_size
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.bridges: List[str] = []   # for stats compatibility
        self.merges: List = []         # for stats compatibility
        self._built = False

    def build(self, pairs: List[Tuple[str, str]]) -> "CharLevelTokenizer":
        vocab: List[str] = list(SPECIAL_TOKENS)
        # Add every character seen in the corpus
        chars = set()
        for nl, smi in pairs:
            chars.update(nl.lower())
            chars.update(smi)
        # Always include SMILES base alphabet
        chars.update(SMILES_BASE_ALPHABET)
        for ch in sorted(chars):
            if ch not in vocab:
                vocab.append(ch)
            if len(vocab) >= self.vocab_size:
                break
        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self._built = True
        return self

    def _encode_stream(self, text: str, stream: str) -> List[int]:
        if stream == "nl":
            tokens = list(text.lower())
        else:
            tokens = list(text)
        unk_id = self.token2id[UNK_TOKEN]
        return [self.token2id.get(t, unk_id) for t in tokens]

    def encode(self, nl_text: str, smiles: str = "") -> Tuple[List[int], List[int]]:
        nl_ids = self._encode_stream(nl_text, "nl")
        smi_ids = self._encode_stream(smiles, "chem") if smiles else []
        return nl_ids, smi_ids

    def encode_full(self, nl_text: str, smiles: str) -> List[int]:
        nl_ids, smi_ids = self.encode(nl_text, smiles)
        return (
            [self.token2id[BOS_TOKEN]]
            + nl_ids
            + [self.token2id[NL_SEP]]
            + smi_ids
            + [self.token2id[EOS_TOKEN]]
        )

    def decode(self, ids: List[int], stream: Optional[str] = None) -> str:
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, UNK_TOKEN)
            if tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def statistics(self, pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        lens = []
        token_counts: Counter = Counter()
        for nl, smi in pairs:
            full = self.encode_full(nl, smi)
            lens.append(len(full))
            for tid in full:
                token_counts[tid] += 1
        total = sum(token_counts.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in token_counts.values()) if total else 0.0
        return {
            "vocab_size": len(self.token2id),
            "num_bridges": 0,
            "num_merges": 0,
            "mean_seq_len": sum(lens) / len(lens) if lens else 0.0,
            "mean_nl_len": 0.0,
            "mean_smi_len": 0.0,
            "max_seq_len": max(lens) if lens else 0,
            "entropy_bits_per_token": entropy,
            "bridge_coverage_pct": 0.0,
            "compression_ratio_vs_char": 1.0,  # baseline by definition
        }

    def __len__(self) -> int:
        return len(self.token2id)

    def __repr__(self) -> str:
        return f"CharLevelTokenizer(vocab={len(self.token2id)})"


# ============================================================ Separate BPE
class _SingleStreamBPE:
    """Helper: a single-stream BPE tokenizer (NL or Chem)."""

    def __init__(self, vocab_size: int, stream: str):
        self.vocab_size = vocab_size
        self.stream = stream  # "nl" or "chem"
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: List[Tuple[Tuple[str, str], str]] = []

    def build(self, texts: List[str]) -> "_SingleStreamBPE":
        # Pre-tokenize
        if self.stream == "nl":
            streams = [nl_pre_tokenize(t) for t in texts]
        else:
            streams = [smiles_pre_tokenize(t) for t in texts]

        # Always include SMILES base alphabet if chem stream
        if self.stream == "chem":
            vocab = list(SMILES_BASE_ALPHABET)
        else:
            vocab = []
        # Add all single tokens from corpus
        base = set()
        for s in streams:
            base.update(s)
        for tok in sorted(base):
            if tok not in vocab:
                vocab.append(tok)

        # BPE merges
        while len(vocab) < self.vocab_size:
            pair_counts = _get_pair_counts(streams)
            if not pair_counts:
                break
            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            new_token = best_pair[0] + best_pair[1]
            if new_token in vocab:
                break
            vocab.append(new_token)
            self.merges.append((best_pair, new_token))
            streams = _merge_pair(streams, best_pair, new_token)

        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        return self

    def encode(self, text: str) -> List[int]:
        if self.stream == "nl":
            tokens = nl_pre_tokenize(text)
        else:
            tokens = smiles_pre_tokenize(text)
        # Apply merges in order
        for pair, new_token in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        unk_id = 0  # fallback
        return [self.token2id.get(t, unk_id) for t in tokens]


class SeparateBPE:
    """Two independent BPE tokenizers (NL + Chem) with separate vocabularies.

    Total vocab = nl_vocab + chem_vocab + special tokens.
    """

    def __init__(
        self,
        total_vocab_size: int = CFG.sep_bpe_vocab_size,
    ):
        self.total_vocab_size = total_vocab_size
        # Split 50/50 between NL and Chem
        # Reserve 5 special tokens at the top
        self.special_size = len(SPECIAL_TOKENS)
        per_stream = (total_vocab_size - self.special_size) // 2
        self.nl_bpe = _SingleStreamBPE(vocab_size=per_stream, stream="nl")
        self.chem_bpe = _SingleStreamBPE(vocab_size=per_stream, stream="chem")

        # Combined vocab: special + nl + chem
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.bridges: List[str] = []
        self.merges: List = []
        self._built = False

    def build(self, pairs: List[Tuple[str, str]]) -> "SeparateBPE":
        nl_texts = [nl for nl, _ in pairs]
        chem_texts = [smi for _, smi in pairs]

        self.nl_bpe.build(nl_texts)
        self.chem_bpe.build(chem_texts)

        # Build combined vocab: special tokens + NL vocab + Chem vocab
        # (prefix NL tokens with "nl:" and chem tokens with "ch:" to avoid collisions)
        combined: List[str] = list(SPECIAL_TOKENS)
        for tok in self.nl_bpe.token2id:
            combined.append(f"nl:{tok}")
        for tok in self.chem_bpe.token2id:
            combined.append(f"ch:{tok}")

        self.token2id = {tok: i for i, tok in enumerate(combined)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        self._built = True
        return self

    def encode(self, nl_text: str, smiles: str = "") -> Tuple[List[int], List[int]]:
        # NL side
        nl_internal = self.nl_bpe.encode(nl_text)
        # Map NL internal ids → combined ids
        # The NL vocab occupies positions [special_size, special_size + len(nl_bpe.token2id))
        nl_offset = self.special_size
        nl_ids = [nid + nl_offset for nid in nl_internal]
        # SMILES side
        if smiles:
            smi_internal = self.chem_bpe.encode(smiles)
            smi_offset = self.special_size + len(self.nl_bpe.token2id)
            smi_ids = [sid + smi_offset for sid in smi_internal]
        else:
            smi_ids = []
        return nl_ids, smi_ids

    def encode_full(self, nl_text: str, smiles: str) -> List[int]:
        nl_ids, smi_ids = self.encode(nl_text, smiles)
        return (
            [self.token2id[BOS_TOKEN]]
            + nl_ids
            + [self.token2id[NL_SEP]]
            + smi_ids
            + [self.token2id[EOS_TOKEN]]
        )

    def decode(self, ids: List[int], stream: Optional[str] = None) -> str:
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, UNK_TOKEN)
            if tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        if not tokens:
            return ""
        # Strip stream prefixes and reconstruct
        if stream == "chem":
            chem_tokens = [t[3:] for t in tokens if t.startswith("ch:")]
            return "".join(chem_tokens)
        else:
            nl_tokens = [t[3:] for t in tokens if t.startswith("nl:")]
            # Use word-boundary marker reconstruction (same as joint tokenizer)
            from joint_tokenizer import WORD_END_MARKER
            joined = "".join(nl_tokens)
            pieces = joined.split(WORD_END_MARKER)
            return " ".join(p for p in pieces if p)

    def statistics(self, pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        lens = []
        token_counts: Counter = Counter()
        for nl, smi in pairs:
            full = self.encode_full(nl, smi)
            lens.append(len(full))
            for tid in full:
                token_counts[tid] += 1
        total = sum(token_counts.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in token_counts.values()) if total else 0.0
        return {
            "vocab_size": len(self.token2id),
            "num_bridges": 0,
            "num_merges": len(self.nl_bpe.merges) + len(self.chem_bpe.merges),
            "mean_seq_len": sum(lens) / len(lens) if lens else 0.0,
            "mean_nl_len": 0.0,
            "mean_smi_len": 0.0,
            "max_seq_len": max(lens) if lens else 0,
            "entropy_bits_per_token": entropy,
            "bridge_coverage_pct": 0.0,
            "compression_ratio_vs_char": 0.0,  # filled in by compare.py
        }

    def __len__(self) -> int:
        return len(self.token2id)

    def __repr__(self) -> str:
        return (
            f"SeparateBPE(vocab={len(self.token2id)}, "
            f"nl_merges={len(self.nl_bpe.merges)}, chem_merges={len(self.chem_bpe.merges)})"
        )


# ============================================================ demo
if __name__ == "__main__":
    from dataset import load_dataset
    pairs = load_dataset(CFG.data_path)
    print(f"Loaded {len(pairs)} pairs\n")

    print("--- CharLevelTokenizer ---")
    char_tok = CharLevelTokenizer()
    char_tok.build(pairs)
    print(f"  {char_tok}")
    stats = char_tok.statistics(pairs)
    print(f"  mean_seq_len = {stats['mean_seq_len']:.2f}")
    print(f"  entropy      = {stats['entropy_bits_per_token']:.2f} bits/tok\n")

    print("--- SeparateBPE ---")
    sep_tok = SeparateBPE()
    sep_tok.build(pairs)
    print(f"  {sep_tok}")
    stats = sep_tok.statistics(pairs)
    print(f"  mean_seq_len = {stats['mean_seq_len']:.2f}")
    print(f"  entropy      = {stats['entropy_bits_per_token']:.2f} bits/tok\n")

    # Round-trip sanity check on a single pair
    nl, smi = pairs[0]
    print(f"Sample pair: NL={nl!r}  SMILES={smi!r}")
    _, char_smi = char_tok.encode("", smi)
    print(f"  Char: encode → {char_smi}  decode → {char_tok.decode(char_smi, stream='chem')!r}")
    _, sep_smi = sep_tok.encode("", smi)
    print(f"  SepBPE: encode → {sep_smi}  decode → {sep_tok.decode(sep_smi, stream='chem')!r}")