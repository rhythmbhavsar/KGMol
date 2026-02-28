"""
test_fusion.py

Evaluate pretrained GINE and Fusion models for logP regression on ZINC dataset.

Requirements:
- torch
- torch_geometric
- scikit-learn

Protocol:
- Load dataset with mol_index
- Load filtered KG molecule embeddings
- Load pretrained GINE and Fusion models
- Extract logP targets (column 1) batch-safe
- Denormalize predictions and targets using train set mean/std
- Compute and print test MAE and RMSE for both models
- Print sanity checks for batch sizes and mol_index min/max
- No retraining, only evaluation
"""

import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from fusion_model import FusionModel
from train_gine import GINEModel

# Load dataset with mol_index
data_list = torch.load("data/raw/zinc_pyg_dataset_with_index.pt")

# Load KG embeddings
kg_embeddings = torch.load("embeddings/kg_embeddings.pt")
kg_molecule_embeddings = kg_embeddings["molecule"]  # shape [num_molecules, kg_dim]
N = kg_molecule_embeddings.shape[0]

# Filter dataset: keep only Data objects with valid mol_index
filtered_data_list = [d for d in data_list if d.mol_index.item() < N]

# Extract logP target exactly like test_gine.py
for data in filtered_data_list:
    if data.y.dim() > 0:
        logp = data.y[1].item()
    else:
        logp = float(data.y)
    data.y = torch.tensor([logp], dtype=torch.float)

# Split dataset (80/10/10)
n_total = len(filtered_data_list)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

indices = np.arange(n_total)
np.random.seed(42)
np.random.shuffle(indices)

train_idx = indices[:n_train]
test_idx = indices[n_train + n_val:]

train_set = [filtered_data_list[i] for i in train_idx]
test_set  = [filtered_data_list[i] for i in test_idx]

# Compute normalization stats from train_set only
def compute_normalization(dataset):
    y = torch.cat([data.y for data in dataset]).view(-1)
    mean = y.mean().item()
    std = y.std().item()
    return mean, std

def denormalize(tensor, mean, std):
    return tensor * std + mean

mean, std = compute_normalization(train_set)
print(f"Target normalization: mean={mean:.6f}, std={std:.6f}")

# Normalize targets for test_set (for evaluation)
for data in test_set:
    data.y = (data.y - mean) / std

# DataLoader
batch_size = 64
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load pretrained GINE model
gine_model = GINEModel(
    node_feat_dim=train_set[0].x.shape[1],
    edge_feat_dim=train_set[0].edge_attr.shape[1],
    num_atom_types=int(torch.max(torch.cat([data.x[:, 0] for data in train_set])).item()) + 1,
    emb_dim=64,
    hidden_dim=256,
    dropout=0.25
).to(device)
gine_model.load_state_dict(torch.load("models/best_gine_model.pt", map_location=device))
gine_model.eval()

# Load pretrained FusionModel
fusion_model = FusionModel(gine_model, kg_molecule_embeddings, freeze_gine=True, mlp_hidden=128).to(device)
fusion_model.load_state_dict(torch.load("models/best_fusion_model.pt", map_location=device))
fusion_model.eval()

# Sanity check: print batch sizes and mol_index min/max
for batch in test_loader:
    print(f"Test batch size: {batch.num_graphs}, mol_index min: {batch.mol_index.min().item()}, max: {batch.mol_index.max().item()}")
    break

# Evaluation function (batch-safe, denormalized)
def evaluate(model, loader, use_kg=False):
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_kg:
                mol_indices = batch.mol_index.to(device)
                valid_mask = mol_indices < kg_molecule_embeddings.shape[0]
                if not torch.all(valid_mask):
                    from torch_geometric.data import Batch
                    batch_list = batch.to_data_list()
                    filtered_batch_list = [batch_list[i] for i, v in enumerate(valid_mask) if v]
                    batch = Batch.from_data_list(filtered_batch_list)
                    mol_indices = mol_indices[valid_mask]
                pred = model(batch, mol_indices)
            else:
                pred = model(batch)
            # Extract logP column
            current_batch_size = batch.num_graphs
            if batch.y.shape[0] == current_batch_size * 5:
                target = batch.y.view(current_batch_size, 5)[:, 1]
            elif len(batch.y.shape) == 2 and batch.y.shape[1] == 5:
                target = batch.y[:, 1]
            else:
                target = batch.y.view(-1)
            # Denormalize
            y_true.append(denormalize(target.cpu(), mean, std))
            y_pred.append(denormalize(pred.cpu(), mean, std))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# Evaluate FusionModel
fusion_mae, fusion_rmse = evaluate(fusion_model, test_loader, use_kg=True)
print(f"Fusion Test MAE: {fusion_mae:.4f}, RMSE: {fusion_rmse:.4f}")

# Corrected GINE baseline evaluation
def evaluate_gine(model, loader, mean, std):
    """
    Evaluate the pretrained GINE baseline model with proper target extraction
    and denormalization, matching Fusion evaluation.
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Predict
            pred = model(batch)

            # Extract logP column (column 1) from targets
            current_batch_size = batch.num_graphs
            if batch.y.shape[0] == current_batch_size * 5:
                target = batch.y.view(current_batch_size, 5)[:, 1]
            elif len(batch.y.shape) == 2 and batch.y.shape[1] == 5:
                target = batch.y[:, 1]
            else:
                target = batch.y.view(-1)

            # Denormalize both predictions and targets
            y_true.append(denormalize(target.cpu(), mean, std))
            y_pred.append(denormalize(pred.cpu(), mean, std))

    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse

# Run corrected evaluation
gine_mae, gine_rmse = evaluate_gine(gine_model, test_loader, mean, std)
print(f"GINE Baseline Test MAE: {gine_mae:.4f}, RMSE: {gine_rmse:.4f}")
print(f"Fusion improvement (MAE): {gine_mae - fusion_mae:.4f}")