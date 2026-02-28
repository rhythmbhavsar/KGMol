import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import random
import numpy as np

# Set fixed random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Device setup: use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Dataset and preprocess logP target
def load_dataset(path):
    # Load list of Data objects
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

# 3. DataLoaders
def get_dataloaders(train_set, val_set, test_set, batch_size=64):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# 4. Model Architecture: 3-layer GCN + 2-layer MLP head
class GCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()  # Ensure features are float for GCNConv
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)  # Pool over nodes in each graph
        out = self.mlp(x)
        return out.view(-1)

# 5. Training function
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

# 6. Evaluation function (MAE, RMSE)
def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y_true.append(batch.y.view(-1).cpu())
            y_pred.append(pred.cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    return mae, rmse

# 7. Main training loop
def main():
    # Load and preprocess dataset
    data_list = load_dataset('src/data/zinc_pyg_dataset.pt')
    train_set, val_set, test_set = split_dataset(data_list)
    train_loader, val_loader, test_loader = get_dataloaders(train_set, val_set, test_set, batch_size=64)

    # Model setup
    in_dim = train_set[0].x.shape[1]
    model = GCNRegressor(in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training
    print("Training baseline GCN for logP regression...")
    for epoch in range(1, 51):
        train_loss = train(model, train_loader, optimizer, loss_fn)
        val_mae, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val MAE = {val_mae:.4f}")

    # Final evaluation
    test_mae, test_rmse = evaluate(model, test_loader)
    print("\nFinal Test Metrics:")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    main()

# ------------------------------
# Script structure:
# - load_dataset: loads and preprocesses logP target
# - split_dataset: splits into train/val/test with fixed seed
# - get_dataloaders: creates DataLoaders
# - GCNRegressor: 3-layer GCN + 2-layer MLP head
# - train: training loop for one epoch
# - evaluate: computes MAE and RMSE
# - main: orchestrates training and evaluation
# ------------------------------
# Comments are provided for clarity. Model is moved to GPU if available.