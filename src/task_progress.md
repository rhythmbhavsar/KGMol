## KGBuilder RuntimeError Fix (2026-02-26)

- Identified and resolved a RuntimeError in `src/kg_builder.py` caused by attempting to call `.item()` on a multi-element tensor (`data.y`).
- Updated code to robustly extract logP from `data.y` using `data.y[1].item()` if `data.y` has more than one element, otherwise `data.y.item()`.
- Verified fix by running the script and confirming successful execution and correct output.
- KG HeteroData object is now generated and saved without error.

Task Progress:
- [x] Analyze requirements (understand the error and context)
- [x] Read src/kg_builder.py around line 95 to see how data.y is used
- [x] Identify the correct way to handle multi-element tensors in this context (logP is at index 1)
- [x] Update the code to fix the error (replace .item() with [1].item() or .item() as appropriate)
- [x] Test the implementation to verify the fix
- [x] Document the change in task_progress.md

---

## PHASE 1 — KG DESIGN (Completed)

### KG Schema (src/kg_builder.py)
- **Nodes**:
  - atom: atomic number
  - functional_group: one-hot group type, size (count), avg logP
  - molecule: avg logP, functional group counts
  - property: placeholder (currently [[0.0]])
- **Edges**:
  - atom_bonded_to_atom
  - atom_part_of_functional_group
  - molecule_contains_functional_group
  - functional_group_correlates_with_property
- **Representation**: PyTorch Geometric HeteroData
- **Status**: Matches requirements, modular, ready for embedding/model phase

---

## PHASE 2 — KG CONSTRUCTION (Completed)

- Functional groups, atom memberships, group stats extracted via RDKit and SMARTS patterns.
- Mapping dictionaries (atom_type_to_id, fg_to_id, mol_to_id) constructed.
- edge_index tensors for all relations built.
- Duplicate nodes avoided, efficient indexing via sets and dictionaries.
- GPU compatibility ensured via device parameter.
- KG object saved to disk.

---

# Molecular Property Prediction Pipeline with KG Fusion

## PHASE 1 — KG DESIGN
- [x] Define KG schema (nodes, edges, features)
- [x] Represent KG as PyTorch Geometric HeteroData

## PHASE 2 — KG CONSTRUCTION
- [x] Extract functional groups, atom memberships, group stats (RDKit)
- [x] Build mapping dictionaries (atom_type, functional_group, molecule_id)
- [x] Construct edge_index tensors per relation
- [x] Ensure no duplicate nodes, efficient indexing, GPU compatibility
- [x] Save KG object to disk

## PHASE 3 — KG EMBEDDING MODEL
- [ ] Implement heterogeneous GNN (HeteroConv/relational GNN)
- [ ] 2–3 layers (GraphSAGE/GAT)
- [ ] Output fixed-size embedding per node
- [ ] Self-supervised training (link prediction) or joint fine-tuning
- [ ] Return embedding lookup table

## PHASE 4 — FUSION WITH GINE MOLECULAR EMBEDDINGS
- [ ] Extract GINE molecular embedding
- [ ] Identify functional groups and atom types per molecule
- [ ] Retrieve KG embeddings
- [ ] Aggregate KG embeddings (mean pooling/attention)
- [ ] Fuse embeddings (concat)
- [ ] Pass fused embedding to MLP regression head

## PHASE 5 — TRAINING PIPELINE
- [ ] Keep original train/val/test splits
- [ ] Train fusion model
- [ ] Use MAE, RMSE, early stopping
- [ ] Compare against baseline GINE-only model
- [ ] Print validation improvement, test metrics, parameter count

## PHASE 6 — OPTIONAL ADVANCED EXTENSIONS
- [ ] Attention weights over functional groups
- [ ] Fine-tune KG embeddings during training
- [ ] Multi-property prediction (logP + solubility)
- [ ] Save final fused model for MoleculeScorer inference

## IMPLEMENTATION REQUIREMENTS
- [ ] Modular file structure: kg_builder.py, kg_model.py, fusion_model.py, train_fusion.py
- [ ] Clean object-oriented design
- [ ] GPU support
- [ ] Deterministic seeds
- [ ] Efficient batching
- [ ] Graceful handling of invalid molecules
- [ ] Complete runnable code