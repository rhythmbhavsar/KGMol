"""
BatchMoleculeScorer: Predict logP for multiple molecules using a trained GINE model.

Features:
- Single or batch SMILES prediction
- Denormalizes predicted logP
- Logs failed SMILES to a separate CSV
- Saves results to CSV
- Uses GPU if available
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm
from rdkit import Chem
import pandas as pd

class BatchMoleculeScorer:
    def __init__(self, model_path, mean, std, node_feat_dim, edge_feat_dim, num_atom_types, emb_dim=64, hidden_dim=256, dropout=0.25):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = mean
        self.std = std

        # Model must match training
        self.model = GINEModel(node_feat_dim, edge_feat_dim, num_atom_types, emb_dim, hidden_dim, dropout).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def smiles_to_pyg(self, smiles):
        """Convert SMILES to PyG Data object; returns None if invalid."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)

        node_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            formal_charge = atom.GetFormalCharge()
            hybridization = int(atom.GetHybridization())
            is_aromatic = int(atom.GetIsAromatic())
            node_features.append([atomic_num, formal_charge, hybridization, is_aromatic])
        x = torch.tensor(node_features, dtype=torch.float)

        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = self.bond_type_to_int(bond.GetBondType())
            is_conjugated = int(bond.GetIsConjugated())
            is_in_ring = int(bond.IsInRing())
            edge_features.append([bond_type, is_conjugated, is_in_ring])
            edge_index.append([i, j])
            edge_features.append([bond_type, is_conjugated, is_in_ring])
            edge_index.append([j, i])

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @staticmethod
    def bond_type_to_int(bond_type):
        mapping = {Chem.BondType.SINGLE:0, Chem.BondType.DOUBLE:1, Chem.BondType.TRIPLE:2, Chem.BondType.AROMATIC:3}
        return mapping.get(bond_type, 0)

    def predict_logp(self, smiles):
        """Predict logP for single SMILES string."""
        data = self.smiles_to_pyg(smiles)
        if data is None or data.x.shape[0] == 0:
            return None
        data = data.to(self.device)
        with torch.no_grad():
            pred = self.model(data)
            logp = pred.item() * self.std + self.mean
        return logp

    def predict_batch(self, smiles_list, save_csv=None):
        """Predict logP for a list of SMILES. Optionally save results to CSV."""
        results = []
        failed = []

        for smi in smiles_list:
            try:
                logp = self.predict_logp(smi)
                if logp is None:
                    failed.append(smi)
                    results.append({'SMILES': smi, 'logP': None})
                else:
                    results.append({'SMILES': smi, 'logP': logp})
            except Exception as e:
                failed.append(smi)
                results.append({'SMILES': smi, 'logP': None})

        df = pd.DataFrame(results)
        if save_csv:
            df.to_csv(save_csv, index=False)
            if failed:
                print(f"Warning: {len(failed)} SMILES failed. Check CSV output for details.")
        return df

# ------------------------------
# GINEModel must match your trained model
class GINEModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, num_atom_types, emb_dim=64, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.atom_emb = nn.Embedding(num_atom_types, emb_dim)
        self.gine1 = GINEConv(nn.Sequential(nn.Linear(emb_dim+node_feat_dim-1, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim)),
                              edge_dim=edge_feat_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.gine2 = GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim)),
                              edge_dim=edge_feat_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.gine3 = GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim)),
                              edge_dim=edge_feat_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.gine4 = GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim)),
                              edge_dim=edge_feat_dim)
        self.bn4 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,1))

    def forward(self, data):
        atomic_number = data.x[:,0].long()
        other_feats = data.x[:,1:].float()
        atom_emb = self.atom_emb(atomic_number)
        x = torch.cat([atom_emb, other_feats], dim=1)
        edge_attr = data.edge_attr.float()
        edge_index = data.edge_index
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for gin, bn in zip([self.gine1,self.gine2,self.gine3,self.gine4],
                           [self.bn1,self.bn2,self.bn3,self.bn4]):
            x = gin(x, edge_index, edge_attr)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.mlp(x).view(-1)

# ------------------------------
# Example usage
if __name__ == "__main__":
    model_path = "best_gine_model.pt"
    mean = 2.825956106185913
    std = 1.1611888408660889
    node_feat_dim = 4
    edge_feat_dim = 3
    num_atom_types = 54

    scorer = BatchMoleculeScorer(model_path, mean, std, node_feat_dim, edge_feat_dim, num_atom_types)

    smiles_list = ["CCO","c1ccccc1","CC(=O)NC1=CC=CC=C1"]
    df_results = scorer.predict_batch(smiles_list, save_csv="batch_logp_predictions.csv")
    print(df_results)