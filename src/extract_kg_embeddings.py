"""
extract_kg_embeddings.py

Extracts molecule node embeddings from src/data/kg_hetero.pt and saves as kg_embeddings.pt for FusionModel.

Requirements:
- torch
- torch_geometric
"""

import torch

hetero = torch.load("embeddings/kg_hetero_src.pt")
molecule_emb = hetero["molecule"].x  # shape [num_molecules, kg_dim]
torch.save({"molecule": molecule_emb}, "embeddings/kg_embeddings.pt")
print(f"Saved molecule embeddings to embeddings/kg_embeddings.pt with shape {molecule_emb.shape}")
