"""
Central configuration for the methodology package.

All hyperparameters live here so you can tune them in one place.
Edit values, then re-run `python main.py` — no other code changes needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ----- Dataset -----
    dataset_seed: int = 0
    n_pairs_per_molecule: int = 20          # 50 molecules × 20 = 1000 pairs

    # ----- Tokenizer (joint) -----
    joint_vocab_size: int = 4096            # Production: 16384. PoC: 4096 (runs in seconds)
    num_bridges: int = 0                    # Experiment 2: no bridges (isolate bridge contribution)
    bridge_min_len: int = 4                 # Minimum SMILES length to be a bridge candidate
    bridge_max_len: int = 12                # Maximum SMILES length
    bridge_min_freq: int = 3                # Must appear at least N times in corpus
    js_threshold: float = 0.15              # Jensen-Shannon divergence gate for cross-stream merges
    js_min_cooccurrence: int = 2            # Min co-occurrence count to compute JS

    # ----- Tokenizer (baselines) -----
    char_vocab_size: int = 256              # Char-level: one token per ASCII char
    sep_bpe_vocab_size: int = 4096          # Separate BPE: same total as joint, split 50/50

    # ----- Verification -----
    verify_n_samples: int = 200             # Lossless round-trip test on N random pairs

    # ----- Paths -----
    data_path: str = "data/synthetic_pairs.json"
    ckpt_dir: str = "ckpt"


# Single global config — import this everywhere
CFG = Config()