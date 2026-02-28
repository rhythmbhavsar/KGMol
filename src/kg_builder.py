"""
kg_builder.py

Builds a heterogeneous chemical Knowledge Graph (KG) from the ZINC PyG dataset.

- Nodes: atom types, functional groups, molecules
- Edges: atom_bonded_to_atom, atom_part_of_functional_group, molecule_contains_functional_group, functional_group_correlates_with_property
- Node features: atomic number, valence, aromaticity, electronegativity (atom); group type, size, frequency (functional group); avg logP, group counts (molecule)
- Output: PyTorch Geometric HeteroData object, saved to disk

Requirements:
- RDKit, PyTorch, PyTorch Geometric
- GPU support
- Efficient indexing, batching
- Graceful handling of invalid molecules
"""

import torch
from torch_geometric.data import HeteroData
from rdkit import Chem
from rdkit.Chem import rdmolops, rdMolDescriptors
import numpy as np
import csv

class KGBuilder:
    def __init__(self, zinc_data_path, smiles_csv_path, save_path, device=None):
        self.zinc_data_path = zinc_data_path
        self.smiles_csv_path = smiles_csv_path
        self.save_path = save_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.atom_types = set()
        self.functional_groups = set()
        self.molecule_ids = []
        self.atom_type_to_id = {}
        self.fg_to_id = {}
        self.mol_to_id = {}
        self.fg_stats = {}
        self.data_list = []
        self.smiles_list = []

    def load_smiles(self):
        self.smiles_list = []
        with open(self.smiles_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.smiles_list.append(row['smiles'])

    def extract_functional_groups(self, mol):
        fg_smarts = {
            "hydroxyl": "[OX2H]", "amine": "[NX3;H2,H1;!$(NC=O)]", "carboxyl": "C(=O)[OH]",
            "aromatic_ring": "c1ccccc1", "carbonyl": "[CX3]=O", "aldehyde": "[CX3H1](=O)[#6]",
            "ketone": "[CX3](=O)[#6]", "ether": "[OD2]([#6])[#6]", "halide": "[F,Cl,Br,I]", "nitro": "[NX3](=O)[O-]"
        }
        groups = set()
        for name, smarts in fg_smarts.items():
            patt = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(patt):
                groups.add(name)
        return groups

    def build(self):
        # Load ZINC PyG dataset
        self.data_list = torch.load(self.zinc_data_path)
        print(f"Loaded {len(self.data_list)} molecules from {self.zinc_data_path}")

        # Load SMILES from CSV
        self.load_smiles()
        print(f"Loaded {len(self.smiles_list)} SMILES from {self.smiles_csv_path}")

        fg_counts = {}
        fg_logp = {}

        # Diagnostic: print first 3 data objects and their attributes
        print("First 3 ZINC Data objects and their attributes:")
        for i in range(min(3, len(self.data_list))):
            print(f"data[{i}]: {self.data_list[i]}")
            print(f"Attributes: {dir(self.data_list[i])}")
            print(f"SMILES: {self.smiles_list[i]}")

        for idx, data in enumerate(self.data_list):
            smiles = self.smiles_list[idx] if idx < len(self.smiles_list) else None
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                continue
            self.molecule_ids.append(idx)
            # Atom types
            for atom in mol.GetAtoms():
                atom_type = atom.GetSymbol()
                self.atom_types.add(atom_type)
            # Functional groups
            groups = self.extract_functional_groups(mol)
            for fg in groups:
                self.functional_groups.add(fg)
                fg_counts[fg] = fg_counts.get(fg, 0) + 1
                logp = data.y[1].item() if data.y.numel() > 1 else data.y.item()
                fg_logp.setdefault(fg, []).append(logp)

        print(f"Collected atom_types: {self.atom_types}")
        print(f"Collected functional_groups: {self.functional_groups}")
        print(f"Collected molecule_ids: {self.molecule_ids[:10]} (showing first 10)")

        # Assign node ids
        self.atom_type_to_id = {a: i for i, a in enumerate(sorted(self.atom_types))}
        self.fg_to_id = {fg: i for i, fg in enumerate(sorted(self.functional_groups))}
        self.mol_to_id = {m: i for i, m in enumerate(self.molecule_ids)}
        self.fg_stats = {fg: {"count": fg_counts.get(fg, 0), "avg_logp": np.mean(fg_logp[fg]) if fg in fg_logp else 0.0} for fg in self.functional_groups}

        # Build HeteroData
        data = HeteroData()
        atom_features = []
        for atom in self.atom_types:
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(atom)
            atom_features.append([atomic_num])
        data['atom'].x = torch.tensor(atom_features, dtype=torch.float, device=self.device)
        fg_features = []
        for fg in self.functional_groups:
            fg_onehot = [int(fg == name) for name in sorted(self.functional_groups)]
            fg_size = self.fg_stats[fg]["count"]
            fg_avg_logp = self.fg_stats[fg]["avg_logp"]
            fg_features.append(fg_onehot + [fg_size, fg_avg_logp])
        data['functional_group'].x = torch.tensor(fg_features, dtype=torch.float, device=self.device)
        mol_features = []
        for m in self.molecule_ids:
            data_obj = self.data_list[m]
            avg_logp = data_obj.y[1].item() if data_obj.y.numel() > 1 else data_obj.y.item()
            fg_count_vec = [0] * len(self.functional_groups)
            smiles = self.smiles_list[m] if m < len(self.smiles_list) else None
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol:
                groups = self.extract_functional_groups(mol)
                for fg in groups:
                    fg_count_vec[self.fg_to_id[fg]] += 1
            mol_features.append([avg_logp] + fg_count_vec)
        data['molecule'].x = torch.tensor(mol_features, dtype=torch.float, device=self.device)

        atom_edges = []
        for m in self.molecule_ids:
            smiles = self.smiles_list[m] if m < len(self.smiles_list) else None
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol:
                for bond in mol.GetBonds():
                    a1 = bond.GetBeginAtom().GetSymbol()
                    a2 = bond.GetEndAtom().GetSymbol()
                    atom_edges.append([self.atom_type_to_id[a1], self.atom_type_to_id[a2]])
        if atom_edges:
            data['atom', 'atom_bonded_to_atom', 'atom'].edge_index = torch.tensor(atom_edges, dtype=torch.long, device=self.device).t().contiguous()
        else:
            data['atom', 'atom_bonded_to_atom', 'atom'].edge_index = torch.empty((2,0), dtype=torch.long, device=self.device)
        atom_fg_edges = []
        for m in self.molecule_ids:
            smiles = self.smiles_list[m] if m < len(self.smiles_list) else None
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol:
                groups = self.extract_functional_groups(mol)
                for atom in mol.GetAtoms():
                    atom_type = atom.GetSymbol()
                    for fg in groups:
                        atom_fg_edges.append([self.atom_type_to_id[atom_type], self.fg_to_id[fg]])
        if atom_fg_edges:
            data['atom', 'atom_part_of_functional_group', 'functional_group'].edge_index = torch.tensor(atom_fg_edges, dtype=torch.long, device=self.device).t().contiguous()
        else:
            data['atom', 'atom_part_of_functional_group', 'functional_group'].edge_index = torch.empty((2,0), dtype=torch.long, device=self.device)
        mol_fg_edges = []
        for m in self.molecule_ids:
            smiles = self.smiles_list[m] if m < len(self.smiles_list) else None
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol:
                groups = self.extract_functional_groups(mol)
                for fg in groups:
                    mol_fg_edges.append([self.mol_to_id[m], self.fg_to_id[fg]])
        if mol_fg_edges:
            data['molecule', 'molecule_contains_functional_group', 'functional_group'].edge_index = torch.tensor(mol_fg_edges, dtype=torch.long, device=self.device).t().contiguous()
        else:
            data['molecule', 'molecule_contains_functional_group', 'functional_group'].edge_index = torch.empty((2,0), dtype=torch.long, device=self.device)
        fg_property_edges = []
        for fg in self.functional_groups:
            fg_property_edges.append([self.fg_to_id[fg], 0])  # property node id = 0
        data['functional_group', 'functional_group_correlates_with_property', 'property'].edge_index = torch.tensor(fg_property_edges, dtype=torch.long, device=self.device).t().contiguous()

        data['property'].x = torch.tensor([[0.0]], dtype=torch.float, device=self.device)

        torch.save(data, self.save_path)
        print(f"KG HeteroData object saved to {self.save_path}")

if __name__ == "__main__":
    builder = KGBuilder(
        zinc_data_path="data/raw/zinc_pyg_dataset.pt",
        smiles_csv_path="data/raw/zinc_smiles_clean.csv",
        save_path="embeddings/kg_hetero_src.pt"
    )
    builder.build()

    print("\n=== KG TEST ===")
    kg = torch.load("embeddings/kg_hetero_src.pt")
    for node_type in kg.node_types:
        x = kg[node_type].x
        print(f"Node type '{node_type}': x shape = {x.shape}")
    for edge_type in kg.edge_types:
        edge_index = kg[edge_type].edge_index
        print(f"Edge type '{edge_type}': edge_index shape = {edge_index.shape}")
    for node_type in kg.node_types:
        x = kg[node_type].x
        print(f"First 3 features for '{node_type}':\n{x[:3]}")
    for edge_type in kg.edge_types:
        edge_index = kg[edge_type].edge_index
        print(f"First 3 edges for '{edge_type}':\n{edge_index[:, :3]}")
    print("=== KG TEST COMPLETE ===")