"""
zinc_csv_to_pyg_dataset.py

Convert cleaned ZINC CSV to PyTorch Geometric dataset (.pt file) for GNN training.

Features:
- Atom features: atomic number, formal charge, hybridization, aromaticity
- Bond features: bond type, conjugation, ring membership
- Robust error handling and logging
- Progress bar for large datasets
- Modular functions for reuse
- Saves PyG Data objects to zinc_pyg_dataset.pt
- Logs failed molecules to zinc_failed.csv

"""

import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import HybridizationType
from tqdm import tqdm

ATOM_HYBRIDIZATION_MAP = {
    HybridizationType.SP: 0,
    HybridizationType.SP2: 1,
    HybridizationType.SP3: 2,
    HybridizationType.SP3D: 3,
    HybridizationType.SP3D2: 4,
    HybridizationType.UNSPECIFIED: 5,
}

BOND_TYPE_MAP = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}

def atom_features(atom):
    """Extract atom features: atomic number, formal charge, hybridization, aromaticity."""
    return [
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        ATOM_HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5),
        int(atom.GetIsAromatic()),
    ]

def bond_features(bond):
    """Extract bond features: bond type, conjugation, ring membership."""
    return [
        BOND_TYPE_MAP.get(bond.GetBondType(), 0),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

def mol_to_pyg_data(mol, y=None):
    """Convert RDKit Mol to PyG Data object."""
    # Node features
    atoms = list(mol.GetAtoms())
    x = torch.tensor([atom_features(atom) for atom in atoms], dtype=torch.long)

    # Edge features and adjacency
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions for undirected graph
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bond_features(bond), bond_features(bond)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    # Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float)
    return data

def process_zinc_csv_to_pyg(
    input_csv="data/raw/zinc_smiles_clean.csv",
    output_pt="data/raw/zinc_pyg_dataset.pt",
    failed_csv="data/raw/zinc_failed.csv",
    include_properties=True
):
    """Main processing function."""
    df = pd.read_csv(input_csv)
    print("DEBUG: df.shape =", df.shape)
    print("DEBUG: df.head():")
    print(df.head())
    if df.shape[0] == 0:
        print("WARNING: DataFrame is empty after reading CSV!")
        return
    print("DEBUG: Entering molecule processing loop...")
    data_list = []
    failed_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        smiles = row.get("smiles")

        if pd.isna(smiles):
            failed_rows.append(row.to_dict())
            continue

        smiles = str(smiles).strip()

        if smiles == "":
            failed_rows.append(row.to_dict())
            continue
        index = row["index"]
        if idx < 5:
            print(f"Index: {index}, SMILES: {smiles}")
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"RDKit MolFromSmiles returned None for index {index}: {smiles}")
                failed_rows.append(row.to_dict())
                continue
            if idx < 5:
                print(f"Created RDKit mol for index {index}: {smiles}, type: {type(mol)}")
        except Exception as e:
            print(f"SMILES conversion failed for index {index}: {smiles}")
            print(f"Exception: {e}")
            failed_rows.append(row.to_dict())
            continue

        # Optional: include molecular properties as y
        y = None
        if include_properties:
            if include_properties:
                props = ["molwt", "logp", "ring_count", "hbd", "hba"]
                if any(pd.isna(row[p]) for p in props):
                    failed_rows.append(row.to_dict())
                    continue
                y = [float(row[p]) for p in props]

        try:
            if idx < 5:
                print("Calling mol_to_pyg_data for first molecule...")
            data = mol_to_pyg_data(mol, y=y)
            data_list.append(data)
        except Exception as e:
            import traceback
            if idx < 5:
                print("Exception traceback for first molecule:")
                traceback.print_exc()
            print(f"Exception: {e}")
            failed_rows.append(row.to_dict())
            continue

    # Save PyG dataset
    torch.save(data_list, output_pt)

    # Save failed molecules
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_csv, index=False)

    # Print summary
    print(f"Saved {len(data_list)} PyG Data objects to {output_pt}")
    print(f"Failed molecules: {len(failed_rows)} (saved to {failed_csv})")

if __name__ == "__main__":
    print("torch_geometric.data.Data class:", Data)
    process_zinc_csv_to_pyg()

"""
Step-by-step explanation:

1. Loads the cleaned ZINC CSV.
2. For each molecule:
   - Converts SMILES to RDKit Mol.
   - Extracts atom features (atomic number, formal charge, hybridization, aromaticity).
   - Extracts bond features (bond type, conjugation, ring membership).
   - Builds PyTorch Geometric Data object (x, edge_index, edge_attr, y).
   - If conversion fails, logs the row to a failed CSV.
3. Saves all Data objects to a .pt file for fast GNN training.
4. Uses a progress bar for large datasets.
5. Modularizes feature extraction for reuse.

**Verification Note:** To verify the resulting PyG graphs, check that:
- The number of atoms and bonds in each Data object matches the original CSV.
- Atom and bond features are consistent with RDKit and the CSV properties.
- The y tensor matches the molecular properties from the CSV.

"""