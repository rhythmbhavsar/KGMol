import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm
import random
import numpy as np

# Set fixed random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device selected: {device} (torch.cuda.is_available() = {torch.cuda.is_available()})")
if device.type == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print(device)

# 1. Load Dataset and preprocess logP target
def load_dataset(path):
    data_list = torch.load(path, weights_only=False)
    # Extract logP (index 1) from y and replace y with scalar tensor
    for data in data_list:
        logp = data.y[1].item() if data.y.dim() > 0 else float(data.y)
        data.y = torch.tensor([logp], dtype=torch.float)
    return data_list

# 2. Dataset Split (80/10/10)
def split_dataset(data_list, seed=SEED):
    n_total = len(data_list)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    indices = np.arange(n_total)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    train_set = [data_list[i] for i in train_idx]
    val_set = [data_list[i] for i in val_idx]
    test_set = [data_list[i] for i in test_idx]
    return train_set, val_set, test_set

# 3. Target Normalization
def compute_normalization(train_set):
    y_train = torch.cat([data.y for data in train_set]).view(-1)
    mean = y_train.mean().item()
    std = y_train.std().item()
    return mean, std

def normalize_targets(data_set, mean, std):
    for data in data_set:
        data.y = (data.y - mean) / std

def denormalize(tensor, mean, std):
    return tensor * std + mean

# 4. DataLoaders
def get_dataloaders(train_set, val_set, test_set, batch_size=64):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class GINEModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, num_atom_types, emb_dim=64, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim  # Expose hidden_dim for embedding shape
        # Atomic number embedding
        self.atom_emb = nn.Embedding(num_atom_types, emb_dim)
        # GINEConv layers
        self.gine1 = GINEConv(
            nn.Sequential(
                nn.Linear(emb_dim + node_feat_dim - 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.bn1 = BatchNorm(hidden_dim)
        self.gine2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.bn2 = BatchNorm(hidden_dim)
        self.gine3 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.bn3 = BatchNorm(hidden_dim)
        self.gine4 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_feat_dim
        )
        self.bn4 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # MLP regression head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, return_embedding=False):
        # Node features: [atomic_number, formal_charge, hybridization, aromaticity]
        atomic_number = data.x[:, 0].long()  # shape [num_nodes]
        other_feats = data.x[:, 1:].float()  # shape [num_nodes, node_feat_dim-1]
        atom_emb = self.atom_emb(atomic_number)  # shape [num_nodes, emb_dim]
        x = torch.cat([atom_emb, other_feats], dim=1)  # shape [num_nodes, emb_dim + node_feat_dim-1]
        edge_attr = data.edge_attr.float()  # shape [num_edges, edge_feat_dim]
        edge_index = data.edge_index
        batch = data.batch

        x = self.gine1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gine2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gine3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.gine4(x, edge_index, edge_attr)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        graph_emb = global_mean_pool(x, batch)
        if return_embedding:
            return graph_emb
        out = self.mlp(graph_emb)
        return out.view(-1)

    def get_embedding(self, data):
        # Returns graph-level embedding (after global_mean_pool, before MLP)
        return self.forward(data, return_embedding=True)

# 6. Training function with early stopping and scheduler
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def evaluate(model, loader, mean, std):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            # Denormalize predictions and targets
            y_true.append(denormalize(batch.y.view(-1).cpu(), mean, std))
            y_pred.append(denormalize(pred.cpu(), mean, std))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    return mae, rmse

# 7. Main training loop with early stopping
def main():
    # Load and preprocess dataset
    data_list = load_dataset('data/raw/zinc_pyg_dataset.pt')
    train_set, val_set, test_set = split_dataset(data_list)
    mean, std = compute_normalization(train_set)
    normalize_targets(train_set, mean, std)
    normalize_targets(val_set, mean, std)
    normalize_targets(test_set, mean, std)
    train_loader, val_loader, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=64)

    # Model setup
    node_feat_dim = train_set[0].x.shape[1]
    edge_feat_dim = train_set[0].edge_attr.shape[1]
    num_atom_types = int(torch.max(torch.cat([data.x[:, 0] for data in train_set])).item()) + 1
    model = GINEModel(node_feat_dim, edge_feat_dim, num_atom_types, emb_dim=64, hidden_dim=256, dropout=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    patience = 10
    best_val_mae = float('inf')
    best_epoch = 0
    epochs = 100
    early_stop_counter = 0
    best_model_state = None

    print("Training edge-aware GINE model for logP regression...")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn)
        val_mae, _ = evaluate(model, val_loader, mean, std)
        scheduler.step(val_mae)
        print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f}, Val MAE = {val_mae:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val MAE: {best_val_mae:.4f} (epoch {best_epoch})")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save best model for future use
        torch.save(best_model_state, "models/best_gine_model.pt")
        print("Best model state_dict saved to models/best_gine_model.pt")
    test_mae, test_rmse = evaluate(model, test_loader, mean, std)
    print("\nFinal Test Metrics:")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    print("Dataset size:", len(data_list))
    print("Train/Val/Test sizes:", len(train_set), len(val_set), len(test_set))
    print("Train mean/std:", mean, std)
    print("First 5 test targets:", [data.y.item() for data in test_set[:5]])
    
if __name__ == "__main__":
    main()

# ------------------------------
# Script structure:
# - load_dataset: loads and preprocesses logP target
# - split_dataset: splits into train/val/test with fixed seed
# - compute_normalization/normalize_targets/denormalize: target normalization utilities
# - get_dataloaders: creates DataLoaders
# - GINEModel: 4-layer GINEConv, atomic number embedding, strong architecture
# - train: training loop for one epoch
# - evaluate: computes MAE and RMSE in original scale
# - main: orchestrates training, early stopping, and evaluation
# ------------------------------
# Comments are provided for clarity. Model is moved to GPU if available.