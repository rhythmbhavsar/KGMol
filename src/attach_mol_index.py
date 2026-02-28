"""
attach_mol_index.py

Script to add mol_index to each PyG Data object in zinc_pyg_dataset.pt.
Assumes you have a mapping from SMILES to KG index (e.g., dict or CSV).

Requirements:
- torch
- torch_geometric
"""

import torch

# Load mapping: dict {smiles: mol_index}
# Example: from CSV with columns 'smiles','mol_index'
import pandas as pd
mapping_csv = "data/raw/zinc_smiles_clean.csv"
mapping_df = pd.read_csv(mapping_csv)
smiles_to_index = dict(zip(mapping_df["smiles"], mapping_df["index"]))

# Load dataset
data_list = torch.load("data/raw/zinc_pyg_dataset.pt")

# Attach mol_index to each Data object
for i, data in enumerate(data_list):
    data.mol_index = torch.tensor(i, dtype=torch.long)
print(f"Number of molecules: {len(data_list)}")
# Save updated dataset
torch.save(data_list, "data/raw/zinc_pyg_dataset_with_index.pt")
print("mol_index attached to all Data objects. Saved to data/raw/zinc_pyg_dataset_with_index.pt.")



import torch
import numpy as np
from torch_geometric.data import DataLoader

# Load dataset with mol_index
data_list = torch.load("data/raw/zinc_pyg_dataset_with_index.pt")

# Load KG embeddings
kg_embeddings = torch.load("embeddings/kg_embeddings.pt")
kg_molecule_embeddings = kg_embeddings["molecule"]  # shape [num_molecules, kg_dim]
N = kg_molecule_embeddings.shape[0]  # number of molecules in KG

# Filter dataset: keep only Data objects with valid mol_index
filtered_data_list = [d for d in data_list if d.mol_index.item() < N]

print(f"Original dataset size: {len(data_list)}")
print(f"Filtered dataset size: {len(filtered_data_list)}")
print(f"Mol_index min: {min(d.mol_index.item() for d in filtered_data_list)}")
print(f"Mol_index max: {max(d.mol_index.item() for d in filtered_data_list)}")

# Split dataset (80/10/10)
n_total = len(filtered_data_list)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

indices = np.arange(n_total)
np.random.seed(42)
np.random.shuffle(indices)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

train_set = [filtered_data_list[i] for i in train_idx]
val_set   = [filtered_data_list[i] for i in val_idx]
test_set  = [filtered_data_list[i] for i in test_idx]

# DataLoaders
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

print(f"Train set: {len(train_set)}, Val set: {len(val_set)}, Test set: {len(test_set)}")

for batch in train_loader:
    print(f"Train batch mol_indices: min={batch.mol_index.min().item()}, max={batch.mol_index.max().item()}")
    break

for batch in val_loader:
    print(f"val_loader batch mol_indices: min={batch.mol_index.min().item()}, max={batch.mol_index.max().item()}")
    break

for batch in test_loader:
    print(f"test_loader batch mol_indices: min={batch.mol_index.min().item()}, max={batch.mol_index.max().item()}")
    break