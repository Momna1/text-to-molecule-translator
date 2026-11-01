"""
Single entry point — runs the entire methodology pipeline.

Usage:
    python main.py                 # run everything
    python main.py --step data     # only generate the dataset
    python main.py --step verify   # only run lossless round-trip test
    python main.py --step compare  # only run the comparison table

Steps (in order):
    1. data     — generate the synthetic dataset (1020 pairs)
    2. verify   — verify the joint tokenizer is lossless (in-vocab + held-out)
    3. compare  — train all 3 tokenizers, print ablation table
    4. demo     — show a live encode/decode demo on a sample pair

Total runtime: ~5 seconds on CPU.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

from config import CFG


def step_data() -> List[Tuple[str, str]]:
    """Step 1: Generate the synthetic dataset."""
    print("\n" + "=" * 70)
    print("STEP 1: Generating synthetic dataset")
    print("=" * 70)
    from dataset import generate_dataset, save_dataset, load_dataset

    if os.path.exists(CFG.data_path):
        print(f"Dataset already exists at {CFG.data_path}, loading...")
        pairs = load_dataset(CFG.data_path)
    else:
        print(f"Generating {50 * CFG.n_pairs_per_molecule} pairs...")
        pairs = generate_dataset(seed=CFG.dataset_seed, n_per_molecule=CFG.n_pairs_per_molecule)
        save_dataset(pairs, CFG.data_path)

    # Verify all SMILES are RDKit-valid (if rdkit is installed)
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdKit.*")
        invalid = [(nl, s) for nl, s in pairs if Chem.MolFromSmiles(s) is None]
        if invalid:
            print(f"WARNING: {len(invalid)} pairs have invalid SMILES")
        else:
            print(f"✓ All {len(pairs)} SMILES are RDKit-valid")
    except ImportError:
        print("(rdkit not installed — skipping SMILES validation)")

    print(f"Sample pair:")
    print(f"  NL    : {pairs[0][0]}")
    print(f"  SMILES: {pairs[0][1]}")
    return pairs


def step_verify(pairs: List[Tuple[str, str]]) -> bool:
    """Step 2: Verify lossless round-trip."""
    print("\n" + "=" * 70)
    print("STEP 2: Lossless round-trip verification")
    print("=" * 70)
    from verify import verify_in_vocab, verify_held_out, HELD_OUT_SMILES
    from joint_tokenizer import JointTokenizer
    import random

    print("Building joint tokenizer...")
    t0 = time.perf_counter()
    tok = JointTokenizer()
    tok.build(pairs)
    print(f"  Built in {time.perf_counter() - t0:.2f}s  →  {tok}")

    random.seed(0)
    print(f"\nIn-vocab round-trip ({CFG.verify_n_samples} samples)...")
    n_pass, n_total, errors = verify_in_vocab(tok, pairs, CFG.verify_n_samples)
    print(f"  Passed: {n_pass}/{n_total}  ({100*n_pass/n_total:.1f}%)")
    if errors:
        for e in errors[:3]:
            print(f"  - {e}")

    print(f"\nHeld-out round-trip ({len(HELD_OUT_SMILES)} novel SMILES)...")
    n_pass_h, n_total_h, errors_h = verify_held_out(tok, HELD_OUT_SMILES)
    print(f"  Passed: {n_pass_h}/{n_total_h}  ({100*n_pass_h/n_total_h:.1f}%)")
    if errors_h:
        for e in errors_h[:3]:
            print(f"  - {e}")

    success = (n_pass == n_total) and (n_pass_h == n_total_h)
    if success:
        print("\n✓ ALL ROUND-TRIP TESTS PASSED — tokenizer is lossless.")
    else:
        print("\n✗ SOME TESTS FAILED — see errors above.")
    return success


def step_compare(pairs: List[Tuple[str, str]]) -> None:
    """Step 3: Train all tokenizers and print comparison table."""
    print("\n" + "=" * 70)
    print("STEP 3: Tokenizer comparison (ablation table)")
    print("=" * 70)
    from compare import train_all, print_table, save_json
    results = train_all(pairs)
    print_table(results)

    os.makedirs("results", exist_ok=True)
    save_json(results, "results/comparison.json")


def step_demo(pairs: List[Tuple[str, str]]) -> None:
    """Step 4: Live demo of encode/decode."""
    print("\n" + "=" * 70)
    print("STEP 4: Live encode/decode demo")
    print("=" * 70)
    from joint_tokenizer import JointTokenizer

    tok = JointTokenizer()
    tok.build(pairs)
    print(f"Tokenizer: {tok}\n")

    # Pick 3 sample pairs (one simple, one medium, one with heterocycle)
    sample_indices = [0, 4, 19]  # aspirin, caffeine, indole-like
    for idx in sample_indices:
        if idx >= len(pairs):
            continue
        nl, smi = pairs[idx]
        print(f"Pair {idx + 1}:")
        print(f"  NL    : {nl}")
        print(f"  SMILES: {smi}")

        nl_ids, smi_ids = tok.encode(nl, smi)
        full_ids = tok.encode_full(nl, smi)
        print(f"  NL ids  ({len(nl_ids):>2}): {nl_ids}")
        print(f"  SMILES ids ({len(smi_ids):>2}): {smi_ids}")
        print(f"  Full ids ({len(full_ids):>2}): {full_ids}")

        decoded_smi = tok.decode(smi_ids, stream="chem")
        decoded_nl = tok.decode(nl_ids, stream="nl")
        print(f"  Decoded SMILES: {decoded_smi}  {'✓' if decoded_smi == smi else '✗'}")
        print(f"  Decoded NL    : {decoded_nl}")
        print()

    # Show the bridge tokens
    print(f"Discovered bridge tokens ({len(tok.bridges)} total):")
    for i, b in enumerate(tok.bridges[:15]):
        print(f"  {i+1:>3}. {b!r}")
    if len(tok.bridges) > 15:
        print(f"  ... and {len(tok.bridges) - 15} more")


def main():
    ap = argparse.ArgumentParser(description="Methodology pipeline — joint tokenizer PoC")
    ap.add_argument(
        "--step",
        choices=["data", "verify", "compare", "demo", "all"],
        default="all",
        help="which step to run (default: all)",
    )
    args = ap.parse_args()

    t_start = time.perf_counter()

    if args.step in ("data", "all"):
        pairs = step_data()
    else:
        from dataset import load_dataset
        if not os.path.exists(CFG.data_path):
            print(f"Dataset not found at {CFG.data_path}. Run `python main.py --step data` first.")
            sys.exit(1)
        pairs = load_dataset(CFG.data_path)

    if args.step in ("verify", "all"):
        success = step_verify(pairs)
        if not success and args.step == "all":
            print("\nVerification failed — aborting subsequent steps.")
            sys.exit(1)

    if args.step in ("compare", "all"):
        step_compare(pairs)

    if args.step in ("demo", "all"):
        step_demo(pairs)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"Done. Total time: {elapsed:.2f}s")
    print(f"{'=' * 70}")
    print(f"\nNext steps:")
    print(f"  - Edit config.py to change hyperparameters")
    print(f"  - Replace data/synthetic_pairs.json with real ChEMBL pairs for production")
    print(f"  - See README.md for the full pipeline architecture")


if __name__ == "__main__":
    main()
