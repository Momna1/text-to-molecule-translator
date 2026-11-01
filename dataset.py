"""
Synthetic ChEMBL-like dataset generator.

Produces 1000 (NL description, SMILES string) pairs by templating 50 common
drug molecules. Realistic enough to:
  - Make bridge tokens emerge naturally (benzene, amide, carboxyl motifs repeat)
  - Test tokenizer flexibility (NL variations per molecule)
  - Verify lossless round-trip on diverse SMILES grammars

For real experiments, replace `generate_dataset()` with a ChEMBL 31 loader
that yields (description, smiles) tuples. The downstream tokenizer code is
dataset-agnostic.

Author: Methodology module — joint tokenizer PoC.
"""
from __future__ import annotations

import json
import os
import random
from typing import List, Tuple

# ----------------------------------------------------------------- molecule DB
# 50 common drug molecules with canonical SMILES and metadata.
# Canonical SMILES verified to be RDKit-parseable.
MOLECULES = [
    # name,              canonical SMILES,                            class,                indication
    ("aspirin",          "CC(=O)Oc1ccccc1C(=O)O",                    "NSAID",              "pain relief"),
    ("paracetamol",      "CC(=O)Nc1ccc(O)cc1",                       "analgesic",          "pain relief"),
    ("ibuprofen",        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",               "NSAID",              "inflammation"),
    ("naproxen",         "COc1cc2cc(C(C)C(=O)O)ccc2cc1",             "NSAID",              "inflammation"),
    ("caffeine",         "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",             "stimulant",          "alertness"),
    ("acetaminophen",    "CC(=O)Nc1ccc(O)cc1",                       "analgesic",          "pain relief"),
    ("salicylic acid",   "O=C(O)c1ccccc1O",                          "NSAID",              "acne treatment"),
    ("phenol",           "Oc1ccccc1",                                "disinfectant",       "antiseptic"),
    ("aniline",          "Nc1ccccc1",                                "precursor",          "chemical synthesis"),
    ("benzoic acid",     "O=C(O)c1ccccc1",                           "preservative",       "antifungal"),
    ("toluene",          "Cc1ccccc1",                                "solvent",            "chemical synthesis"),
    ("benzamide",        "NC(=O)c1ccccc1",                           "intermediate",       "chemical synthesis"),
    ("acetanilide",      "CC(=O)Nc1ccccc1",                          "intermediate",       "chemical synthesis"),
    ("catechol",         "Oc1ccccc1O",                               "intermediate",       "chemical synthesis"),
    ("resorcinol",       "Oc1cccc(O)c1",                             "intermediate",       "chemical synthesis"),
    ("hydroquinone",     "Oc1ccc(O)cc1",                             "intermediate",       "chemical synthesis"),
    ("indole",           "c1ccc2[nH]ccc2c1",                         "intermediate",       "chemical synthesis"),
    ("pyridine",         "c1ccncc1",                                 "solvent",            "chemical synthesis"),
    ("imidazole",        "c1[nH]cnc1",                               "intermediate",       "chemical synthesis"),
    ("furan",            "c1ccoc1",                                  "solvent",            "chemical synthesis"),
    ("thiophene",        "c1ccsc1",                                  "intermediate",       "chemical synthesis"),
    ("pyrrole",          "c1cc[nH]c1",                               "intermediate",       "chemical synthesis"),
    ("nicotinic acid",   "OC(=O)c1cccnc1",                           "vitamin",            "nutrition"),
    ("isoniazid",        "NC(=O)c1ccc(N)cc1",                        "antibiotic",         "tuberculosis"),
    ("metronidazole",    "O=[N+]([O-])c1cc(CO)ncc1Cl",               "antibiotic",         "infection"),
    ("sulfanilamide",    "Nc1ccc(S(=O)(=O)N)cc1",                    "antibiotic",         "infection"),
    ("isonicotinamide",  "NC(=O)c1ccncc1",                           "intermediate",       "chemical synthesis"),
    ("barbituric acid",  "O=C1NC(=O)NC(=O)N1",                       "intermediate",       "chemical synthesis"),
    ("uracil",           "O=c1cc[nH]c(=O)[nH]1",                     "nucleobase",         "biochemistry"),
    ("cytosine",         "Nc1cc[nH]c(=O)n1",                         "nucleobase",         "biochemistry"),
    ("adenine",          "Nc1ncnc2[nH]cnc12",                        "nucleobase",         "biochemistry"),
    ("guanine",          "Nc1[nH]c(=O)c2[nH]cnc2n1",                 "nucleobase",         "biochemistry"),
    ("xanthine",         "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",             "intermediate",       "biochemistry"),
    ("hippuric acid",    "O=C(O)CNc1ccccc1",                         "intermediate",       "biochemistry"),
    ("anthranilic acid", "Nc1ccccc1C(=O)O",                          "intermediate",       "chemical synthesis"),
    ("methyl salicylate","COC(=O)c1ccccc1O",                         "liniment",           "pain relief"),
    ("ethyl benzoate",   "CCOC(=O)c1ccccc1",                         "intermediate",       "chemical synthesis"),
    ("benzyl alcohol",   "OCc1ccccc1",                               "preservative",       "antiseptic"),
    ("acetophenone",     "CC(=O)c1ccccc1",                           "intermediate",       "chemical synthesis"),
    ("benzaldehyde",     "O=Cc1ccccc1",                              "intermediate",       "chemical synthesis"),
    ("nitrobenzene",     "O=[N+]([O-])c1ccccc1",                     "intermediate",       "chemical synthesis"),
    ("chlorobenzene",    "Clc1ccccc1",                               "solvent",            "chemical synthesis"),
    ("bromobenzene",     "Brc1ccccc1",                               "intermediate",       "chemical synthesis"),
    ("anisol",           "COc1ccccc1",                               "solvent",            "chemical synthesis"),
    ("phenetole",        "CCOc1ccccc1",                              "solvent",            "chemical synthesis"),
    ("styrene",          "C=Cc1ccccc1",                              "monomer",            "polymer synthesis"),
    ("cinnamic acid",    "OC(=O)/C=C/c1ccccc1",                      "intermediate",       "chemical synthesis"),
    ("coumarin",         "O=c1ccc2ccccc2o1",                         "fragrance",          "perfumery"),
    ("vanillin",         "COc1cc(C=O)ccc1O",                         "flavorant",          "food industry"),
    ("aspartame",        "COC(=O)Cc1ccccc1NC(=O)Cc2ccccc2N",         "sweetener",          "food industry"),
    ("saccharin",        "O=S1(=O)c2ccccc2C(=O)N1",                  "sweetener",          "food industry"),
]

# ----------------------------------------------------------------- NL templates
# Each template produces a different natural-language description for the
# same molecule. 20 templates × 50 molecules = 1000 pairs.
NL_TEMPLATES = [
    "generate {name}",
    "synthesize {name}",
    "produce {name}",
    "make {name}, a {drug_class}",
    "generate {name} for {indication}",
    "create a {drug_class} for {indication}, specifically {name}",
    "design {name}, an aromatic compound",
    "the molecule {name}, used for {indication}",
    "{name} — a {drug_class} used in {indication}",
    "generate a {drug_class} with the structure of {name}",
    "construct {name}, a {drug_class} containing a benzene ring" ,
    "draw {name}, used as {indication} agent",
    "produce the {drug_class} {name}",
    "create {name}, a compound for {indication}",
    "I need {name}, a {drug_class} used in {indication}",
    "synthesize the {drug_class} {name} for {indication}",
    "the {drug_class} {name} for {indication}",
    "build {name} — a {drug_class} targeting {indication}",
    "{name} — {drug_class} for {indication}",
    "give me {name}, an aromatic {drug_class}",
]


def generate_dataset(seed: int = 0, n_per_molecule: int = 20) -> List[Tuple[str, str]]:
    """Generate (nl_description, smiles) pairs by templating molecules.

    Args:
        seed: random seed for reproducibility
        n_per_molecule: number of NL variations per molecule

    Returns:
        List of (nl, smiles) tuples, length = len(MOLECULES) * n_per_molecule
    """
    rng = random.Random(seed)
    pairs = []
    for name, smi, drug_class, indication in MOLECULES:
        templates = rng.sample(NL_TEMPLATES, min(n_per_molecule, len(NL_TEMPLATES)))
        for tmpl in templates:
            nl = tmpl.format(
                name=name,
                drug_class=drug_class,
                indication=indication,
            )
            pairs.append((nl, smi))
    rng.shuffle(pairs)
    return pairs


def save_dataset(pairs: List[Tuple[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [{"nl": nl, "smiles": smi} for nl, smi in pairs],
            f, ensure_ascii=False, indent=2,
        )
    print(f"Saved {len(pairs)} pairs to {path}")


def load_dataset(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(d["nl"], d["smiles"]) for d in data]


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "data", "synthetic_pairs.json")
    pairs = generate_dataset(seed=0, n_per_molecule=20)
    save_dataset(pairs, out_path)

    # Sanity check: verify all SMILES are RDKit-valid
    try:
        from rdkit import Chem
        invalid = [(nl, smi) for nl, smi in pairs if Chem.MolFromSmiles(smi) is None]
        if invalid:
            print(f"WARNING: {len(invalid)} pairs have invalid SMILES:")
            for nl, smi in invalid[:5]:
                print(f"  {smi}  ←  {nl}")
        else:
            print(f"All {len(pairs)} SMILES are RDKit-valid.")
    except ImportError:
        print("rdkit not installed — skipping SMILES validation.")

    # Print sample
    print("\nFirst 5 pairs:")
    for nl, smi in pairs[:5]:
        print(f"  NL   : {nl}")
        print(f"  SMILES: {smi}\n")
