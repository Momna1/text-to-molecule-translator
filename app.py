"""
Streamlit GUI for the text-to-molecule pipeline.

Run with:
    streamlit run app.py

A browser tab will open at http://localhost:8501 with:
- A text input for your natural-language prompt
- A "Generate Molecule" button
- The generated SMILES + validity check
- A property table (MW, LogP, TPSA, QED, rings, rotatable bonds)
- A 2D molecule structure image
- Latency display
- An expander showing the tokenized prompt + generated tokens
- An optional reference SMILES box for Tanimoto similarity

Author: GUI for the methodology PoC.
"""
from __future__ import annotations

import io
import os
import time
from typing import Optional

# Silence RDKit logging before any imports
import logging
logging.getLogger("rdkit").setLevel(logging.CRITICAL)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdKit.*")
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

import streamlit as st
import torch

from joint_tokenizer import JointTokenizer
from model import DecoderConfig, LightweightDecoder, quantize_int8
from generate import MoleculeGenerator


# ============================================================ page config
st.set_page_config(
    page_title="Text-to-Molecule Translator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================ cached model loader
@st.cache_resource(show_spinner=False)
def load_generator() -> MoleculeGenerator:
    """Load the trained model + tokenizer. Cached so it only loads once."""
    ckpt_dir = "ckpt"
    if not (os.path.exists(os.path.join(ckpt_dir, "tokenizer.json"))
            and os.path.exists(os.path.join(ckpt_dir, "model.pt"))):
        return None
    try:
        return MoleculeGenerator.from_checkpoint(ckpt_dir, use_int8=True)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ============================================================ header
st.title("Ultra-Low Latency Text-to-Molecule Translation")
st.markdown(
    "*Bridging human and chemical languages via optimized AI tokenization*"
)
st.markdown("---")

# ============================================================ sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This is a proof-of-concept of the text-to-molecule pipeline described
        in the research article.

        **Pipeline:**
        1. NL prompt → joint BPE tokenizer
        2. Tokenized prompt → lightweight Transformer decoder
        3. Generated tokens → SMILES string
        4. SMILES → RDKit validation + property extraction
        5. Valid molecule → 2D structure image

        **Note:** PoC model trained on 1020 pairs (~30 seconds on CPU).
        For real chemistry, train on ChEMBL 31 (1.9M pairs).
        """
    )

    st.header("Generation parameters")
    temperature = st.slider(
        "Temperature (higher = more random)",
        min_value=0.1, max_value=2.0, value=0.85, step=0.05,
    )
    top_p = st.slider(
        "Top-p (nucleus sampling threshold)",
        min_value=0.1, max_value=1.0, value=0.9, step=0.05,
    )
    max_tokens = st.slider(
        "Max tokens to generate",
        min_value=16, max_value=200, value=80, step=8,
    )

    st.header("Model status")
    if "generator" not in st.session_state:
        st.session_state.generator = load_generator()
    gen = st.session_state.generator
    if gen is None:
        st.warning("No trained model found. Run `python train.py` first.")
    else:
        st.success(f"Model loaded (vocab={len(gen.tokenizer)})")


# ============================================================ main panel
col_input, col_image = st.columns([3, 2])

with col_input:
    st.subheader("Natural-Language Prompt")
    prompt = st.text_input(
        "Enter your instruction:",
        value="generate a painkiller with a benzene ring",
        help="Examples: 'synthesize aspirin', 'create paracetamol', 'make caffeine'",
    )

    reference_smiles = st.text_input(
        "Reference SMILES (optional, for Tanimoto similarity):",
        value="",
        help="Leave blank to skip similarity comparison.",
    )

    generate_btn = st.button("Generate Molecule", type="primary", use_container_width=True)


# ============================================================ generation logic
if generate_btn and gen is not None:
    with st.spinner("Generating molecule..."):
        result = gen.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reference_smiles=reference_smiles if reference_smiles else None,
        )
        st.session_state.last_result = result
        st.session_state.last_prompt = prompt


# ============================================================ display results
if "last_result" in st.session_state:
    result = st.session_state.last_result

    st.markdown("---")
    st.subheader("Result")

    if result.valid:
        st.success(f"Valid molecule generated in {result.latency_ms:.1f} ms")
    else:
        st.error(f"Invalid SMILES generated in {result.latency_ms:.1f} ms (post-processing failed)")

    col_props, col_extra = st.columns([1, 1])

    with col_props:
        if result.valid:
            st.markdown(f"**Generated SMILES:**")
            st.code(result.canonical_smiles, language="smiles")

            st.markdown("**Physico-chemical properties:**")
            props_data = {
                "Property": [
                    "Molecular weight (Da)",
                    "LogP (lipophilicity)",
                    "TPSA (Å²)",
                    "QED (drug-likeness)",
                    "Number of rings",
                    "Rotatable bonds",
                    "Tokens generated",
                    "Latency (ms)",
                ],
                "Value": [
                    f"{result.molecular_weight:.2f}",
                    f"{result.logp:.2f}",
                    f"{result.tpsa:.2f}",
                    f"{result.qed:.3f}",
                    f"{result.num_rings}",
                    f"{result.num_rotatable_bonds}",
                    f"{result.num_tokens_generated}",
                    f"{result.latency_ms:.1f}",
                ],
            }
            st.table(props_data)

            if result.tanimoto_to_reference is not None:
                st.markdown(f"**Tanimoto similarity to reference:** {result.tanimoto_to_reference:.3f}")
        else:
            st.markdown(f"**Raw generated output:**")
            st.code(result.raw_smiles, language="text")
            st.markdown(f"**Repair attempted:** {result.repair_attempted}")

    with col_extra:
        with st.expander("Tokenization details", expanded=False):
            nl_ids, _ = gen.tokenizer.encode(prompt, smiles="")
            st.markdown(f"**Prompt tokens ({len(nl_ids)}):**")
            st.code(str(nl_ids), language="text")

            token_strs = [gen.tokenizer.id2token.get(i, "<unk>") for i in nl_ids]
            st.markdown("**Token strings:**")
            st.code(" | ".join(token_strs), language="text")

        if result.valid:
            st.markdown("**2D structure:**")
            try:
                from rdkit import Chem
                from rdkit.Chem import Draw
                mol = Chem.MolFromSmiles(result.canonical_smiles)
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(400, 400))
                    st.image(img, caption=result.canonical_smiles, use_container_width=True)

                    os.makedirs("results", exist_ok=True)
                    img.save("results/generated_molecule.png")
            except Exception as e:
                st.warning(f"Could not render 2D structure: {e}")


# ============================================================ footer
st.markdown("---")
st.markdown(
    """
    <small>
    PoC pipeline for the research article <i>Ultra-Low Latency Text-to-Molecule
    Translation: Bridging Human and Chemical Languages via Optimized AI Tokenization</i>.
    Supervisor: Dr. Nasir. Draft v1.0.
    </small>
    """,
    unsafe_allow_html=True,
)