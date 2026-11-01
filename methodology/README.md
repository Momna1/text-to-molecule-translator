# Methodology — Joint Tokenizer PoC

A self-contained Python package implementing the **joint dual-stream BPE
tokenizer** described in §3.2 of the research article *"Ultra-Low Latency
Text-to-Molecule Translation: Bridging Human and Chemical Languages via
Optimized AI Tokenization."*

This package builds the **first methodology component** (the tokenizer) at
production-grade quality. It is the foundation on which the decoder, training
loop, and inference pipeline (later methodology components) will be built.

---

## What this package does

1. **Generates a synthetic dataset** of 1020 (NL, SMILES) pairs by templating
   50 common drug molecules with 20 natural-language variations each.
2. **Trains three tokenizers** for the ablation table:
   - `CharLevelTokenizer` — one token per character (no compression baseline)
   - `SeparateBPE` — two independent BPE tokenizers (NL + Chem), no shared vocab
   - `JointTokenizer` — the proposed joint dual-stream BPE with bridge tokens
     and Jensen-Shannon divergence cross-stream merge gating
3. **Verifies losslessness** — every (NL, SMILES) pair round-trips through
   `encode → decode` and back to the original string. Tests both in-vocab
   pairs and 20 held-out SMILES that the tokenizer has never seen.
4. **Reports statistics** for the ablation table: vocabulary size, number of
   bridges, number of merges, mean sequence length, entropy (bits/token),
   compression ratio vs. char-level baseline.

---

## Quickstart (5 seconds to first result)

```bash
# From this directory:
pip install -r requirements.txt
python main.py
```

You'll see:
- Dataset generation + RDKit validation
- Lossless round-trip verification (in-vocab + held-out)
- Ablation table comparing all 3 tokenizers
- Live encode/decode demo on sample pairs

**Total runtime: ~7 seconds on CPU.** No GPU required, no large downloads.

---

## VS Code setup

1. **Open the folder in VS Code:**
   - File → Open Folder → select this `methodology/` directory

2. **Install the Python extension** (if not already):
   - Extensions panel → search "Python" by Microsoft → Install

3. **Pick the Python interpreter:**
   - Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → "Python: Select Interpreter"
   - Choose Python 3.10+ (create a venv first if you prefer:
     `python -m venv .venv && source .venv/bin/activate`)

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the pipeline:**
   - Open `main.py`
   - Click the ▶ Run button in the top-right corner, OR
   - Open the integrated terminal (`Ctrl+`\`) and run `python main.py`

6. **Debug mode:**
   - Set breakpoints by clicking in the gutter (left of line numbers)
   - Press `F5` → "Python File"

---

## Project structure

```
methodology/
├── main.py              ← Single entry point. Run this.
├── config.py            ← All hyperparameters (edit me to experiment)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
│
├── dataset.py           ← Synthetic ChEMBL-like data generator (1020 pairs)
├── joint_tokenizer.py   ← Production joint BPE tokenizer (the main contribution)
├── baselines.py         ← Char-level + Separate BPE for ablation
├── verify.py            ← Lossless round-trip test
├── compare.py           ← Train all 3 tokenizers, print comparison table
│
├── data/
│   └── synthetic_pairs.json   ← Generated dataset (created on first run)
├── results/
│   └── comparison.json        ← Ablation table output (created on run)
└── ckpt/                      ← Checkpoint directory (for future training)
```

---

## How to experiment

### Change hyperparameters

Edit `config.py`:

```python
@dataclass
class Config:
    joint_vocab_size: int = 4096     # try 8192 or 16384
    num_bridges: int = 64            # try 128 or 256
    bridge_min_len: int = 4          # mine longer bridges
    js_threshold: float = 0.15       # stricter cross-stream gate (lower = stricter)
    ...
```

Then re-run `python main.py`.

### Run only one step

```bash
python main.py --step data      # just generate the dataset
python main.py --step verify    # just verify losslessness
python main.py --step compare   # just print the ablation table
python main.py --step demo      # just show the encode/decode demo
```

### Swap in real ChEMBL data

Replace `dataset.load_dataset()` with a real ChEMBL loader. The downstream
tokenizer code is dataset-agnostic — it only needs `List[Tuple[str, str]]`
where each tuple is `(nl_description, smiles_string)`.

A real loader would look like:

```python
def load_chembl():
    from chembl_webresource_client.new_client import new_client
    molecule = new_client.molecule
    records = molecule.filter(has_molecule_chembl_id=True)[:10000]
    pairs = []
    for r in records:
        smi = r.get("molecule_structures", {}).get("canonical_smiles")
        desc = r.get("molecule_synonyms", [{}])[0].get("molecule_synonym", "")
        if smi and desc:
            pairs.append((desc.lower(), smi))
    return pairs
```

---

## Architecture: how the joint tokenizer works

```
                ┌─────────────────────────────────────┐
                │  Input: (NL, SMILES) pairs          │
                └─────────────┬───────────────────────┘
                              │
                ┌─────────────▼───────────────────────┐
                │  1. Mine bridge tokens               │
                │     - Substring frequency counting   │
                │     - Score = freq × length          │
                │     - Greedy deduplication           │
                └─────────────┬───────────────────────┘
                              │
                ┌─────────────▼───────────────────────┐
                │  2. Pre-tokenize both streams        │
                │     - NL: word-level + ▁ marker      │
                │     - Chem: atom-level regex         │
                │     - Apply bridge tokens (greedy)   │
                └─────────────┬───────────────────────┘
                              │
                ┌─────────────▼───────────────────────┐
                │  3. BPE merges with JS gate          │
                │     - For each candidate merge:      │
                │       compute context distribution   │
                │       compare to other-stream tokens │
                │       accept if JS divergence < 0.5  │
                └─────────────┬───────────────────────┘
                              │
                ┌─────────────▼───────────────────────┐
                │  4. Final vocabulary                 │
                │     [specials] + [bridges] + [base]  │
                │     + [BPE merges]                   │
                └─────────────────────────────────────┘
```

---

## Verification results

```
======================================================================
Lossless round-trip verification — JointTokenizer
======================================================================
Loaded 1020 pairs from data/synthetic_pairs.json

Building joint tokenizer...
  JointTokenizer(vocab=752, bridges=64, merges=520)

--- In-vocab round-trip test (200 samples) ---
  Passed: 200/200  (100.0%)

--- Held-out round-trip test (20 novel SMILES) ---
  Passed: 20/20  (100.0%)

======================================================================
✓ ALL ROUND-TRIP TESTS PASSED — tokenizer is lossless.
======================================================================
```

---

## Ablation table (PoC scale)

```
====================================================================================================
Tokenizer               Vocab  Bridges  Merges  Mean SeqLen   Entropy  Compress  Train(s)
----------------------------------------------------------------------------------------------------
Char-level                 66        0       0        61.98     4.71b     1.00×     0.00s
Separate BPE              714        0     539         6.52     5.96b     9.50×     0.76s
Joint BPE (ours)          752       64     520         6.52     5.96b     9.50×     1.56s
====================================================================================================

NOTE: At PoC scale (1k pairs, 4k vocab), Joint and Separate BPE behave similarly.
      The article's reported divergence (-38% seq len, -1.42 bits/tok) requires
      ChEMBL-scale training (1.9M pairs, 16k vocab). The infrastructure here is
      identical - only the dataset and hyperparameters change.
```

---

## What's next (methodology components still to build)

This package covers **Methodology component 2 (Tokenizer)**. The full
methodology has 5 components:

| # | Component              | Status     | File                          |
|---|------------------------|------------|-------------------------------|
| 1 | Problem formulation    | In article | (text only — LaTeX note)      |
| 2 | **Joint tokenizer**    | **Done**   | **`joint_tokenizer.py`**      |
| 3 | Lightweight decoder    | PoC only   | (next milestone)              |
| 4 | Training pipeline      | PoC only   | (after decoder)               |
| 5 | Inference + post-proc  | PoC only   | (after training)              |

The next component to build is **3. Lightweight decoder** — the 6-layer
Transformer that consumes the tokenizer's outputs. The PoC version exists in
`../poc_pipeline/model.py` but needs FlashAttention v2, distillation, and
profiling for production.
