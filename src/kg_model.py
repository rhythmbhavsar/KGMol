"""
kg_model.py

Heterogeneous GNN for embedding the chemical Knowledge Graph (KG).

- Uses PyTorch Geometric HeteroConv
- Supports GraphSAGE and GAT layers
- 2–3 layer configurable
- Outputs fixed-size embedding per node type
- Self-supervised training (link prediction) or joint fine-tuning
- Returns embedding lookup table for atom, functional group, fragment, molecule nodes

Requirements:
- torch
- torch_geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv
from torch_geometric.data import HeteroData

class HeteroGNN(nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_dim=128, num_layers=2, conv_type="sage"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.metadata = metadata

        # Input projection layers for each node type
        self.input_proj = nn.ModuleDict()
        for node_type in metadata['node_types']:
            self.input_proj[node_type] = nn.Linear(in_channels_dict[node_type], hidden_dim)

        # Build layer-wise HeteroConv
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in metadata['edge_types']:
                src_type, _, dst_type = edge_type
                src_dim = hidden_dim  # After projection, all node types have hidden_dim features
                dst_dim = hidden_dim
                if conv_type == "sage":
                    conv_dict[edge_type] = SAGEConv(
                        (src_dim, dst_dim),
                        hidden_dim
                    )
                elif conv_type == "gat":
                    conv_dict[edge_type] = GATConv(
                        (src_dim, dst_dim),
                        hidden_dim,
                        heads=1
                    )
                else:
                    raise ValueError("conv_type must be 'sage' or 'gat'")
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # Node embedding heads
        self.emb_heads = nn.ModuleDict()
        for node_type in metadata['node_types']:
            self.emb_heads[node_type] = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {node_type: tensor}
        # edge_index_dict: {edge_type: tensor}
        # Project all node features to hidden_dim
        orig_x_dict = {k: v for k, v in x_dict.items()}  # Save original for node counts
        x_dict = {k: self.input_proj[k](v) for k, v in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # Ensure all node types are present and not None
            for k in self.metadata['node_types']:
                if k not in x_dict or x_dict[k] is None:
                    num_nodes = orig_x_dict[k].shape[0]
                    x_dict[k] = torch.zeros((num_nodes, self.hidden_dim), device=next(self.parameters()).device)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        # Final embedding heads
        emb_dict = {k: self.emb_heads[k](v) for k, v in x_dict.items()}
        return emb_dict

def get_metadata(hetero_data):
    return {
        "node_types": list(hetero_data.node_types),
        "edge_types": list(hetero_data.edge_types)
    }

def get_in_channels_dict(hetero_data):
    return {nt: hetero_data[nt].x.shape[1] for nt in hetero_data.node_types}

def train_link_prediction(model, data, epochs=10, lr=1e-3, device="cuda"):
    # Self-supervised link prediction (example: atom_bonded_to_atom)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    data = data.to(device)
    for epoch in range(epochs):
        model.train()
        emb_dict = model(data.x_dict, data.edge_index_dict)
        # Example: atom_bonded_to_atom link prediction
        edge_index = data["atom", "atom_bonded_to_atom", "atom"].edge_index
        src_emb = emb_dict["atom"][edge_index[0]]
        dst_emb = emb_dict["atom"][edge_index[1]]
        pos_score = (src_emb * dst_emb).sum(dim=1)
        # Negative sampling: sample a fixed number of negatives per epoch
        num_neg = min(10000, emb_dict["atom"].shape[0])
        neg_index = torch.randint(0, emb_dict["atom"].shape[0], (2, num_neg), device=device)
        neg_src = emb_dict["atom"][neg_index[0]]
        neg_dst = emb_dict["atom"][neg_index[1]]
        neg_score = (neg_src * neg_dst).sum(dim=1)
        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} | LinkPred Loss: {loss.item():.4f}")
    # Return embedding lookup table
    model.eval()
    with torch.no_grad():
        emb_dict = model(data.x_dict, data.edge_index_dict)
    return emb_dict

if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(device)
    kg = torch.load("embeddings/kg_hetero_src.pt")
    metadata = get_metadata(kg)
    in_channels_dict = get_in_channels_dict(kg)
    model = HeteroGNN(metadata, in_channels_dict, hidden_dim=128, num_layers=2, conv_type="sage")
    emb_dict = train_link_prediction(model, kg, epochs=3, lr=1e-3, device=device)
    for node_type, emb in emb_dict.items():
        print(f"{node_type} embedding shape: {emb.shape}")
    # Save model weights
    torch.save(model.state_dict(), "models/best_kg_model.pt")
    print("KG model weights saved to models/best_kg_model.pt")
