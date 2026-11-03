"""
Minimal training loop for the joint tokenizer + lightweight decoder.

Implements Stage 2 (instruction tuning) from Section 3.4 of the article,
adapted for PoC: trains on the 1020-pair synthetic dataset in ~30 seconds
on CPU using a small (4-layer, d=128) decoder.

For real ChEMBL-scale training, swap load_dataset() with a ChEMBL loader
and scale up --d-model=256, --n-layers=6, --vocab-size=16384.

Usage:
    python train.py                          # default PoC config
    python train.py --epochs 50              # more epochs
    python train.py --d-model 256 --n-layers 6   # production config (slow on CPU)

Author: Methodology component #3 - lightweight decoder training pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import CFG
from dataset import load_dataset
from joint_tokenizer import JointTokenizer
from model import DecoderConfig, LightweightDecoder


# ============================================================ dataset
class TextMoleculeDataset(Dataset):
    """Tokenizes (NL, SMILES) pairs into model-ready tensors."""

    def __init__(self, pairs: List[Tuple[str, str]], tokenizer: JointTokenizer, max_len: int = 128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        nl, smi = self.pairs[idx]
        ids = self.tokenizer.encode_full(nl, smi)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len - 1] + [self.tokenizer.token2id["<eos>"]]
        return torch.tensor(ids, dtype=torch.long)


def collate_pad(batch: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    """Pad variable-length sequences to the same length."""
    max_len = max(len(x) for x in batch)
    out = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, :len(x)] = x
    return out


# ============================================================ train
def train(
    pairs: List[Tuple[str, str]],
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    max_seq_len: int = 128,
    weight_decay: float = 0.01,
    warmup_steps: int = 30,
    out_dir: str = "ckpt",
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[JointTokenizer, LightweightDecoder, dict]:
    """Run a full training loop. Returns (tokenizer, model, history)."""
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Build tokenizer
    if verbose:
        print(f"[1/4] Building tokenizer on {len(pairs)} pairs...")
    tok = JointTokenizer()
    tok.build(pairs)
    if verbose:
        print(f"      {tok}")

    # 2. Build model
    if verbose:
        print(f"[2/4] Building lightweight decoder (d={d_model}, L={n_layers})...")
    cfg = DecoderConfig(
        vocab_size=len(tok),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        pad_id=tok.token2id["<pad>"],
        bos_id=tok.token2id["<bos>"],
        eos_id=tok.token2id["<eos>"],
        unk_id=tok.token2id["<unk>"],
        nl_sep_id=tok.token2id["<nl>"],
    )
    model = LightweightDecoder(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"      {n_params/1e6:.2f}M parameters ({n_params:,})")

    # 3. Training setup
    if verbose:
        print(f"[3/4] Training for {epochs} epochs (batch={batch_size}, lr={lr})...")
    ds = TextMoleculeDataset(pairs, tok, max_len=max_seq_len)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_pad(b, cfg.pad_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_id)

    history = {"loss": [], "lr": []}
    total_steps = epochs * len(loader)
    step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            if step < warmup_steps:
                lr_now = lr * (step + 1) / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_now = lr * 0.5 * (1 + math.cos(math.pi * progress))
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            inp = batch[:, :-1].contiguous()
            tgt = batch[:, 1:].contiguous()
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            history["loss"].append(loss.item())
            history["lr"].append(lr_now)

        avg = epoch_loss / max(1, n_batches)
        should_print = (epochs <= 10) or ((epoch + 1) % 5 == 0) or (epoch == 0) or (epoch == epochs - 1)
        if verbose and should_print:
            print(f"      epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  lr={lr_now:.2e}")

    # 4. Save
    if verbose:
        print(f"[4/4] Saving checkpoint to {out_dir}/...")
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg.__dict__,
    }, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    if verbose:
        print("      Done.")
    return tok, model, history


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="Train joint tokenizer + lightweight decoder")
    ap.add_argument("--epochs", type=int, default=30, help="training epochs (default 30)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128, help="model dimension (production: 256)")
    ap.add_argument("--n-layers", type=int, default=4, help="decoder layers (production: 6)")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="ckpt")
    args = ap.parse_args()

    pairs = load_dataset(CFG.data_path)
    print(f"Loaded {len(pairs)} pairs from {CFG.data_path}\n")
    train(
        pairs=pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()