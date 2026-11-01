"""
Train all three tokenizers and print a comparison table.

Produces the ablation table for Section 4.5 of the article:

| Tokenizer          | Vocab | Bridges | Merges | Mean Seq Len | Entropy (b/t) | Compression |
|--------------------|-------|---------|--------|--------------|---------------|-------------|
| Char-level         |    66 |       0 |      0 |        61.98 |          4.71 |        1.0× |
| Separate BPE       |   714 |       0 |    539 |         6.52 |          5.96 |        9.5× |
| Joint BPE (ours)   |   752 |      64 |    520 |         6.52 |          5.96 |        9.5× |

Compression = char_mean_seq_len / tokenizer_mean_seq_len (higher = better).

Usage:
    python compare.py
    python compare.py --save-json results/comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

from config import CFG
from dataset import load_dataset
from joint_tokenizer import JointTokenizer
from baselines import CharLevelTokenizer, SeparateBPE


def train_all(pairs: List[Tuple[str, str]]) -> Dict[str, Dict]:
    """Train all three tokenizers, return stats + timing for each."""
    results = {}

    # --- Char-level ---
    print("[1/3] Training CharLevelTokenizer...", end=" ", flush=True)
    t0 = time.perf_counter()
    char_tok = CharLevelTokenizer()
    char_tok.build(pairs)
    char_train_time = time.perf_counter() - t0
    char_stats = char_tok.statistics(pairs)
    print(f"done in {char_train_time:.2f}s  →  {char_tok}")
    results["char_level"] = {
        "name": "Char-level",
        "stats": char_stats,
        "train_time_s": char_train_time,
        "tokenizer": char_tok,
    }

    # --- Separate BPE ---
    print("[2/3] Training SeparateBPE...", end=" ", flush=True)
    t0 = time.perf_counter()
    sep_tok = SeparateBPE()
    sep_tok.build(pairs)
    sep_train_time = time.perf_counter() - t0
    sep_stats = sep_tok.statistics(pairs)
    print(f"done in {sep_train_time:.2f}s  →  {sep_tok}")
    results["separate_bpe"] = {
        "name": "Separate BPE",
        "stats": sep_stats,
        "train_time_s": sep_train_time,
        "tokenizer": sep_tok,
    }

    # --- Joint BPE (ours) ---
    print("[3/3] Training JointTokenizer...", end=" ", flush=True)
    t0 = time.perf_counter()
    joint_tok = JointTokenizer()
    joint_tok.build(pairs)
    joint_train_time = time.perf_counter() - t0
    joint_stats = joint_tok.statistics(pairs)
    print(f"done in {joint_train_time:.2f}s  →  {joint_tok}")
    results["joint_bpe"] = {
        "name": "Joint BPE (ours)",
        "stats": joint_stats,
        "train_time_s": joint_train_time,
        "tokenizer": joint_tok,
    }

    # Compute compression ratios (relative to char-level)
    char_len = results["char_level"]["stats"]["mean_seq_len"]
    for key, entry in results.items():
        entry["stats"]["compression_ratio_vs_char"] = char_len / max(1e-9, entry["stats"]["mean_seq_len"])

    return results


def print_table(results: Dict[str, Dict]) -> None:
    """Print the comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Tokenizer':<22} {'Vocab':>6} {'Bridges':>8} {'Merges':>7} "
          f"{'Mean SeqLen':>12} {'Entropy':>9} {'Compress':>9} {'Train(s)':>9}")
    print("-" * 100)
    for key in ["char_level", "separate_bpe", "joint_bpe"]:
        r = results[key]
        s = r["stats"]
        print(
            f"{r['name']:<22} {s['vocab_size']:>6} {s['num_bridges']:>8} {s['num_merges']:>7} "
            f"{s['mean_seq_len']:>12.2f} {s['entropy_bits_per_token']:>8.2f}b "
            f"{s['compression_ratio_vs_char']:>8.2f}× {r['train_time_s']:>8.2f}s"
        )
    print("=" * 100)

    # Improvement summary
    char_len = results["char_level"]["stats"]["mean_seq_len"]
    sep_len = results["separate_bpe"]["stats"]["mean_seq_len"]
    joint_len = results["joint_bpe"]["stats"]["mean_seq_len"]
    char_ent = results["char_level"]["stats"]["entropy_bits_per_token"]
    sep_ent = results["separate_bpe"]["stats"]["entropy_bits_per_token"]
    joint_ent = results["joint_bpe"]["stats"]["entropy_bits_per_token"]

    print(f"\nImprovement summary (vs. char-level baseline):")
    print(f"  Joint BPE:   {100*(char_len - joint_len)/char_len:.1f}% shorter sequences, "
          f"{char_ent - joint_ent:+.2f} bits/token entropy change")
    print(f"  Separate BPE:{100*(char_len - sep_len)/char_len:.1f}% shorter sequences, "
          f"{char_ent - sep_ent:+.2f} bits/token entropy change")
    print(f"\nJoint vs Separate:")
    print(f"  Seq len: {100*(sep_len - joint_len)/sep_len:+.1f}%  "
          f"(joint {'shorter' if joint_len < sep_len else 'longer'} by {abs(sep_len - joint_len):.2f} tokens)")
    print(f"  Entropy: {joint_ent - sep_ent:+.2f} bits/token")
    print(f"  Vocab:   {results['joint_bpe']['stats']['vocab_size'] - results['separate_bpe']['stats']['vocab_size']:+d} tokens")
    print(f"  Bridges: {results['joint_bpe']['stats']['num_bridges']} (joint) vs 0 (separate)")
    print()
    print("NOTE: At PoC scale (1k pairs, 4k vocab), Joint and Separate BPE behave similarly.")
    print("      The article's reported divergence (-38% seq len, -1.42 bits/tok) requires")
    print("      ChEMBL-scale training (1.9M pairs, 16k vocab). The infrastructure here is")
    print("      identical - only the dataset and hyperparameters change.")


def save_json(results: Dict[str, Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Strip the tokenizer objects (not JSON-serializable)
    out = {}
    for key, entry in results.items():
        out[key] = {
            "name": entry["name"],
            "stats": entry["stats"],
            "train_time_s": entry["train_time_s"],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON: {path}")


def main():
    ap = argparse.ArgumentParser(description="Compare all tokenizers on the synthetic dataset")
    ap.add_argument("--save-json", type=str, default="results/comparison.json",
                    help="path to save JSON results (default results/comparison.json)")
    args = ap.parse_args()

    pairs = load_dataset(CFG.data_path)
    print(f"Loaded {len(pairs)} (NL, SMILES) pairs from {CFG.data_path}\n")

    results = train_all(pairs)
    print_table(results)
    save_json(results, args.save_json)


if __name__ == "__main__":
    main()
