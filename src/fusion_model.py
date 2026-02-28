"""
fusion_model.py

Batch-safe Fusion model for molecular property prediction:
- Combines GINE molecular embedding with molecule-level KG embeddings
- Concatenates GINE and KG embeddings for each molecule in batch
- Passes fused embedding to MLP regression head

Requirements:
- torch
- torch_geometric
"""

import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, gine_model, kg_molecule_embeddings, freeze_gine=True, mlp_hidden=128):
        super().__init__()
        self.gine_model = gine_model
        # Register molecule-level KG embeddings as buffer (not trainable)
        self.register_buffer("kg_molecule_embeddings", kg_molecule_embeddings)
        self.kg_dim = kg_molecule_embeddings.shape[1]
        # Infer GINE embedding dim from model or allow override
        if hasattr(gine_model, "dim"):
            self.gine_dim = gine_model.dim
        elif hasattr(gine_model, "hidden_dim"):
            self.gine_dim = gine_model.hidden_dim
        else:
            self.gine_dim = 128  # Default/fallback
        mlp_input_dim = self.kg_dim + self.gine_dim
        # print(f"FusionModel MLP input dim: {mlp_input_dim} (kg_dim={self.kg_dim}, gine_dim={self.gine_dim})")
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
        if freeze_gine:
            for param in self.gine_model.parameters():
                param.requires_grad = False

    def forward(self, data, mol_indices):
        # data: PyG batch
        # mol_indices: tensor [batch_size], maps to row in kg_molecule_embeddings
        # print("KG molecule embeddings shape===========================:", self.kg_molecule_embeddings.shape)
        gine_emb = self.gine_model.get_embedding(data)  # [batch_size, gine_dim]
        assert torch.all(mol_indices < self.kg_molecule_embeddings.shape[0]), f"Out-of-bounds mol_indices: {mol_indices[~(mol_indices < self.kg_molecule_embeddings.shape[0])]}"
        kg_emb = self.kg_molecule_embeddings[mol_indices]  # [batch_size, kg_dim]
        # print(f"gine_emb shape: {gine_emb.shape}")
        # print(f"kg_emb shape: {kg_emb.shape}")
        fused_emb = torch.cat([gine_emb, kg_emb], dim=1)  # [batch_size, kg_dim + gine_dim]
        # print(f"fused_emb shape: {fused_emb.shape}")
        # print(f"mol_indices shape: {mol_indices.shape}")
        # print(f"fused_emb stats: min={fused_emb.min().item()}, max={fused_emb.max().item()}, mean={fused_emb.mean().item()}, std={fused_emb.std().item()}")
        pred = self.mlp(fused_emb)  # [batch_size, 1]
        return pred.view(-1)

if __name__ == "__main__":
    # Example usage (mock)
    class DummyGINE(nn.Module):
        def __init__(self, dim=128):
            super().__init__()
            self.dim = dim
        def get_embedding(self, data):
            # Mock batch embedding
            return torch.randn(data.num_graphs, self.dim, device=data.x.device)
    # Mock KG molecule embeddings
    kg_molecule_embeddings = torch.randn(1000, 128)
    gine_model = DummyGINE()
    model = FusionModel(gine_model, kg_molecule_embeddings, freeze_gine=True)
    # Mock data and mol_indices
    class DummyData:
        x = torch.randn(20, 4)
        num_graphs = 20
    data = DummyData()
    mol_indices = torch.randint(0, 1000, (20,))
    pred = model(data, mol_indices)
    print(f"Predicted logP batch: {pred.shape}")