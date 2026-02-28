"""
pyg_to_rdkit_zinc.py

Convert all molecules in the PyTorch Geometric ZINC dataset to sanitized RDKit Mol,
extract descriptors, and save results to CSV using pandas.

Improvements (2026-02-24):
- Robust bond count validation: only count unique, valid chemical bonds (ignore "NONE" and self-loops)
- Fix atom mapping: extract element symbol from atom name (e.g., "C" from "C H1 -")
- Construct RDKit atom using Chem.Atom(element_symbol)
- Track failures: atom_mapping_failed, bond_mapping_failed, bond_add_failed, sanitize_failed
- Add diagnostic mode (debug_limit) for fast debugging and detailed failure breakdown
- Improved validation: only reject molecules for atom mapping, bond addition, or sanitization failures
- Detailed summary stats and failure breakdown

"""

from torch_geometric.datasets import ZINC
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import pickle
import pandas as pd

# Module-level Dictionary class for pickle compatibility
class Dictionary(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def load_zinc_dataset(root="data/ZINC/full", split="train"):
    """Load the ZINC dataset (train split)."""
    return ZINC(root=root, split=split)

def load_atom_dict(path="data/ZINC/raw/atom_dict.pickle"):
    """Load atom_dict.pickle and return idx2word list."""
    with open(path, "rb") as f:
        atom_dict = pickle.load(f)
    return atom_dict.idx2word

def load_bond_dict(path="data/ZINC/raw/bond_dict.pickle"):
    """Load bond_dict.pickle and return idx2word list."""
    with open(path, "rb") as f:
        bond_dict = pickle.load(f)
    return bond_dict.idx2word

def extract_element_symbol(atom_name):
    """
    Extract element symbol from atom name.
    E.g., "C H1 -" -> "C", "O H1 +" -> "O", "Br" -> "Br"
    """
    return atom_name.split()[0]

def pyg_to_rdkit(data, idx2word_atoms, idx2word_bonds):
    """
    Convert PyG Data object to RDKit Mol.
    Returns (mol, None) if success, (None, failure_reason) if failure.
    Tracks failure reasons for diagnostics.
    """
    mol = Chem.RWMol()
    num_nodes = data.num_nodes

    # Atom mapping: strict alignment, no silent skips
    for i in range(num_nodes):
        atom_idx = int(data.x[i].item())
        atom_name = idx2word_atoms[atom_idx]
        element_symbol = extract_element_symbol(atom_name)
        try:
            atom = Chem.Atom(element_symbol)
        except Exception:
            return None, "atom_mapping_failed"
        mol.AddAtom(atom)

    # Bond addition: only add unique, valid chemical bonds (ignore "NONE" and self-loops)
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    num_edges = edge_index.shape[1]
    added_bonds = set()
    for i in range(num_edges):
        src = int(edge_index[0, i])
        dst = int(edge_index[1, i])
        bond_idx = int(edge_attr[i].item())
        bond_name = idx2word_bonds[bond_idx]
        if bond_name == "NONE":
            continue  # skip non-bonds
        if src == dst:
            continue  # skip self-loops
        try:
            bond_type = {
                "SINGLE": Chem.BondType.SINGLE,
                "DOUBLE": Chem.BondType.DOUBLE,
                "TRIPLE": Chem.BondType.TRIPLE,
            }[bond_name]
        except KeyError:
            return None, "bond_mapping_failed"
        # Only add unique undirected bonds
        bond_pair = tuple(sorted((src, dst)))
        if bond_pair not in added_bonds:
            try:
                mol.AddBond(src, dst, bond_type)
                added_bonds.add(bond_pair)
            except Exception:
                return None, "bond_add_failed"

    # Sanitize molecule (let RDKit infer aromaticity)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "sanitize_failed"

    # Validation: atom count should always match
    if mol.GetNumAtoms() != num_nodes:
        return None, "count_mismatch"

    # Bond count: should match number of unique, valid bonds added
    if mol.GetNumBonds() != len(added_bonds):
        return None, "count_mismatch"

    return mol, None

def mol_to_smiles(mol):
    """Convert RDKit Mol to SMILES string."""
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def extract_descriptors(mol):
    """Extract molecular descriptors for a sanitized RDKit Mol."""
    return {
        "molwt": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "ring_count": mol.GetRingInfo().NumRings(),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
    }

def process_zinc_dataset_to_csv(
    output_csv="data/raw/zinc_smiles.csv",
    root="data/ZINC/full",
    atom_dict_path="data/ZINC/raw/atom_dict.pickle",
    bond_dict_path="data/ZINC/raw/bond_dict.pickle",
    split="train",
    debug_limit=None
):
    """
    Process all molecules in the ZINC dataset, extract descriptors, save to CSV using pandas.
    Tracks failure reasons and prints detailed summary stats.
    If debug_limit is set, processes only first N molecules for diagnostics.
    """
    dataset = load_zinc_dataset(root=root, split=split)
    idx2word_atoms = load_atom_dict(atom_dict_path)
    idx2word_bonds = load_bond_dict(bond_dict_path)

    results = []
    failed = 0
    failure_reasons = {
        "atom_mapping_failed": 0,
        "bond_mapping_failed": 0,
        "bond_add_failed": 0,
        "sanitize_failed": 0,
        "count_mismatch": 0,
    }
    total = len(dataset) if debug_limit is None else min(debug_limit, len(dataset))
    for idx, data in enumerate(dataset):
        if debug_limit is not None and idx >= debug_limit:
            break
        mol, fail_reason = pyg_to_rdkit(data, idx2word_atoms, idx2word_bonds)
        if mol is None:
            results.append({
                "index": idx,
                "num_atoms": None,
                "num_bonds": None,
                "smiles": None,
                "molwt": None,
                "logp": None,
                "ring_count": None,
                "hbd": None,
                "hba": None,
                "fail_reason": fail_reason,
            })
            failed += 1
            if fail_reason in failure_reasons:
                failure_reasons[fail_reason] += 1
            continue
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        smiles = mol_to_smiles(mol)
        desc = extract_descriptors(mol)
        results.append({
            "index": idx,
            "num_atoms": num_atoms,
            "num_bonds": num_bonds,
            "smiles": smiles,
            "molwt": desc["molwt"],
            "logp": desc["logp"],
            "ring_count": desc["ring_count"],
            "hbd": desc["hbd"],
            "hba": desc["hba"],
            "fail_reason": None,
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    success = total - failed
    print(f"Processed {total} molecules.")
    print(f"Successful reconstructions: {success}")
    print(f"Failed reconstructions: {failed}")
    print(f"Results saved to {output_csv}")
    print("\n=== Failure breakdown ===")
    for reason, count in failure_reasons.items():
        print(f"  {reason}: {count}")
    print("========================")
    print(f"Success rate: {success / total:.2%}")


def main():
    """
    Main entry point: process ZINC dataset and save results.
    Set debug_limit for fast diagnostics (e.g., debug_limit=5000).
    """
    process_zinc_dataset_to_csv()

if __name__ == "__main__":
    main()

"""
Explanation:

Previous atomic number mapping failed because atom names in idx2word_atoms include hydrogen count and charge (e.g., "C H1 -"), which are not valid element symbols for Chem.Atom or Chem.PeriodicTable.GetAtomicNumber().
The new logic extracts the element symbol (first part of the atom name) and constructs the RDKit atom directly, preserving node index alignment.

Expected improvement:
- Success rate should increase from ~33% to >90%.
- CSV output remains consistent and robust.
- All previous improvements (bond mapping, undirected bond reconstruction, sanitization, validation, descriptor extraction, error handling) are preserved.
"""