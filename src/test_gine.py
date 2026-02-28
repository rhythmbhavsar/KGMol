"""
test_gine_baseline.py

Purpose:
- Load pretrained GINE model
- Load dataset (with mol_index if needed)
- Extract logP target exactly like training
- Normalize/denormalize consistently
- Evaluate MAE & RMSE
"""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from train_gine import GINEModel  # assuming your original GINE code is here
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load dataset ---
data_list = torch.load("data/raw/zinc_pyg_dataset_with_index.pt")

# --- Extract logP target exactly like training ---
for data in data_list:
    if data.y.dim() > 0:
        logp = data.y[1].item()
    else:
        logp = float(data.y)
    data.y = torch.tensor([logp], dtype=torch.float)

# --- Split dataset (80/10/10) ---
n_total = len(data_list)
n_train = int(0.8 * n_total)
n_val   = int(0.1 * n_total)
n_test  = n_total - n_train - n_val

indices = np.arange(n_total)
np.random.seed(42)
np.random.shuffle(indices)

train_set = [data_list[i] for i in indices[:n_train]]
val_set   = [data_list[i] for i in indices[n_train:n_train+n_val]]
test_set  = [data_list[i] for i in indices[n_train+n_val:]]

# --- Normalization stats from train_set ---
y_train = torch.cat([d.y for d in train_set]).view(-1)
mean = y_train.mean().item()
std = y_train.std().item()

# Normalize targets
for dataset in [train_set, val_set, test_set]:
    for data in dataset:
        data.y = (data.y - mean) / std

# --- DataLoaders ---
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# --- Load pretrained GINE ---
node_feat_dim = train_set[0].x.shape[1]
edge_feat_dim = train_set[0].edge_attr.shape[1]
num_atom_types = int(torch.max(torch.cat([data.x[:, 0] for data in train_set])).item()) + 1

gine_model = GINEModel(node_feat_dim, edge_feat_dim, num_atom_types, emb_dim=64, hidden_dim=256, dropout=0.25)
gine_model.load_state_dict(torch.load("models/best_gine_model.pt", map_location=device))
gine_model.to(device)
gine_model.eval()

# --- Evaluation function ---
def denormalize(tensor, mean, std):
    return tensor * std + mean

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)  # Use regression head
            target = batch.y.view(-1)
            y_true.append(denormalize(target.cpu(), mean, std))
            y_pred.append(denormalize(pred.cpu(), mean, std))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# --- Run test ---
test_mae, test_rmse = evaluate(gine_model, test_loader)
print(f"Original GINE Baseline Test Metrics:")
print(f"MAE:  {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")