"""
train_fusion.py

Train Fusion model using the exact protocol as test_gine.py:
- Extract logP target as scalar for each data
- 80/10/10 train/val/test split
- Compute normalization stats from train_set
- Normalize targets for train/val/test
- Batch-safe DataLoader
- Train Fusion model (GINE frozen) on train_set, validate on val_set
- Evaluate Fusion and GINE baseline on test_set
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from fusion_model import FusionModel
from train_gine import GINEModel

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Load dataset ---
data_list = torch.load("data/raw/zinc_pyg_dataset_with_index.pt")

# --- Extract logP target exactly like test_gine.py ---
for data in data_list:
    if data.y.dim() > 0:
        logp = data.y[1].item()
    else:
        logp = float(data.y)
    data.y = torch.tensor([logp], dtype=torch.float)

# --- Load KG embeddings ---
kg_embeddings = torch.load("embeddings/kg_embeddings.pt")
kg_molecule_embeddings = kg_embeddings["molecule"]  # shape [num_molecules, kg_dim]
N = kg_molecule_embeddings.shape[0]

# --- Filter dataset: keep only Data objects with valid mol_index ---
filtered_data_list = [d for d in data_list if d.mol_index.item() < N]

# --- Split dataset (80/10/10) ---
n_total = len(filtered_data_list)
n_train = int(0.8 * n_total)
n_val   = int(0.1 * n_total)
n_test  = n_total - n_train - n_val

indices = np.arange(n_total)
np.random.seed(42)
np.random.shuffle(indices)

train_set = [filtered_data_list[i] for i in indices[:n_train]]
val_set   = [filtered_data_list[i] for i in indices[n_train:n_train+n_val]]
test_set  = [filtered_data_list[i] for i in indices[n_train+n_val:]]

# --- Normalization stats from train_set ---
y_train = torch.cat([d.y for d in train_set]).view(-1)
mean = y_train.mean().item()
std = y_train.std().item()
print(f"Target normalization: mean={mean:.6f}, std={std:.6f}")

# --- Normalize targets for train/val/test ---
for dataset in [train_set, val_set, test_set]:
    for data in dataset:
        data.y = (data.y - mean) / std

# --- DataLoaders ---
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# --- Load pretrained GINE model ---
node_feat_dim = train_set[0].x.shape[1]
edge_feat_dim = train_set[0].edge_attr.shape[1]
num_atom_types = int(torch.max(torch.cat([data.x[:, 0] for data in train_set])).item()) + 1

gine_model = GINEModel(node_feat_dim, edge_feat_dim, num_atom_types, emb_dim=64, hidden_dim=256, dropout=0.25).to(device)
gine_model.load_state_dict(torch.load("models/best_gine_model.pt", map_location=device))
gine_model.eval()

# --- Build FusionModel (GINE frozen by default) ---
fusion_model = FusionModel(gine_model, kg_molecule_embeddings, freeze_gine=True, mlp_hidden=128).to(device)

# --- Optimizer (only trainable params) ---
optimizer = optim.Adam(filter(lambda p: p.requires_grad, fusion_model.parameters()), lr=1e-3)
loss_fn = nn.MSELoss()
patience = 10
best_val_mae = float("inf")
patience_counter = 0
best_model_state = None

def denormalize(tensor, mean, std):
    return tensor * std + mean

def evaluate(model, loader, use_kg=False):
    model.eval()
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
            target = batch.y.view(-1)
            y_true.append(denormalize(target.cpu(), mean, std))
            y_pred.append(denormalize(pred.cpu(), mean, std))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def train():
    global best_val_mae, patience_counter, best_model_state
    for epoch in range(1, 101):
        fusion_model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            mol_indices = batch.mol_index.to(device)
            valid_mask = mol_indices < kg_molecule_embeddings.shape[0]
            if not torch.all(valid_mask):
                from torch_geometric.data import Batch
                batch_list = batch.to_data_list()
                filtered_batch_list = [batch_list[i] for i, v in enumerate(valid_mask) if v]
                batch = Batch.from_data_list(filtered_batch_list)
                mol_indices = mol_indices[valid_mask]
            optimizer.zero_grad()
            pred = fusion_model(batch, mol_indices)
            target = batch.y.view(-1)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        avg_loss = total_loss / len(train_loader.dataset)
        # Evaluate on val set
        val_mae, val_rmse = evaluate(fusion_model, val_loader, use_kg=True)
        print(f"Epoch {epoch:03d}: Train Loss = {avg_loss:.4f}, Val MAE = {val_mae:.4f}, Val RMSE = {val_rmse:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = fusion_model.state_dict()
            patience_counter = 0
            torch.save(best_model_state, "models/best_fusion_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model for final evaluation
    if best_model_state is not None:
        fusion_model.load_state_dict(torch.load("models/best_fusion_model.pt", map_location=device))
    fusion_mae, fusion_rmse = evaluate(fusion_model, test_loader, use_kg=True)
    print(f"Final Fusion Test MAE: {fusion_mae:.4f}, RMSE: {fusion_rmse:.4f}")

    # GINE baseline evaluation (no KG, no out-of-bounds filtering)
    gine_mae, gine_rmse = evaluate(gine_model, test_loader, use_kg=False)
    print(f"GINE Baseline Test MAE: {gine_mae:.4f}, RMSE: {gine_rmse:.4f}")
    print(f"Fusion improvement (MAE): {gine_mae - fusion_mae:.4f}")

if __name__ == "__main__":
    # Sanity check: print batch sizes and mol_index min/max
    for batch in train_loader:
        print(f"Train batch size: {batch.num_graphs}, mol_index min: {batch.mol_index.min().item()}, max: {batch.mol_index.max().item()}")
        break
    for batch in val_loader:
        print(f"Val batch size: {batch.num_graphs}, mol_index min: {batch.mol_index.min().item()}, max: {batch.mol_index.max().item()}")
        break
    for batch in test_loader:
        print(f"Test batch size: {batch.num_graphs}, mol_index min: {batch.mol_index.min().item()}, max: {batch.mol_index.max().item()}")
        break
    train()

# ------------------------------
# Comments:
# - Fusion model is trained using the exact protocol as test_gine.py
# - LogP extraction, normalization, splits, and evaluation are fully aligned
# - Out-of-bounds mol_index filtering is applied for Fusion batches only
# - GINE baseline is evaluated on the same test set, no filtering
# - Fully batch-based, no per-molecule loops
# ------------------------------