# Molecular AI Pipeline

This repository provides a complete pipeline for building, embedding, and training on chemical knowledge graphs using the ZINC dataset, PyTorch Geometric, and RDKit. The project is organized for clarity, reproducibility, and modularity.

## Project Structure

```
├── src/                # All Python scripts (training, testing, KG builder/model, data processing)
├── data/
│   ├── raw/           # Raw datasets, CSVs, pickles, .pt files
│   ├── processed/     # Processed datasets, .pt files
├── models/            # Model checkpoints (GINE, Fusion, KG)
├── embeddings/        # KG embeddings (.pt files)
├── notebooks/         # Jupyter notebooks for data exploration and analysis
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
```

## Setup

1. **Create a virtual environment (recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   - Requires Python 3.8+.
   - Key packages: torch, torch-geometric, rdkit, pandas, scikit-learn, tqdm

3. **Directory structure:**  
   All scripts expect data, models, and embeddings to be outside `src/`.  
   - Place raw ZINC files in `data/raw/`
   - Processed files in `data/processed/`
   - Model checkpoints in `models/`
   - KG embeddings in `embeddings/`

## Data Preparation

- **Clean ZINC CSV:**  
  ```
  python src/clean_zinc_smiles_csv.py
  ```
  Cleans the raw ZINC CSV and saves to `data/raw/zinc_smiles_clean.csv`.

- **Convert CSV to PyG Dataset:**  
  ```
  python src/zinc_csv_to_pyg_dataset.py
  ```
  Converts cleaned CSV to PyTorch Geometric dataset (`data/raw/zinc_pyg_dataset.pt`).

- **Attach mol_index:**  
  ```
  python src/attach_mol_index.py
  ```
  Adds molecule index to each PyG Data object.

## Knowledge Graph Construction

- **Build KG:**  
  ```
  python src/kg_builder.py
  ```
  Builds a heterogeneous chemical KG and saves to `embeddings/kg_hetero_src.pt`.

- **Extract KG Embeddings:**  
  ```
  python src/extract_kg_embeddings.py
  ```
  Extracts molecule node embeddings from KG and saves to `embeddings/kg_embeddings.pt`.

## Model Training & Evaluation

- **Train GINE Model:**  
  ```
  python src/train_gine.py
  ```
  Trains the GINE model for logP regression. Checkpoints saved to `models/best_gine_model.pt`.

- **Test GINE Model:**  
  ```
  python src/test_gine.py
  ```
  Evaluates the GINE model on the test set.

- **Train KG Model:**  
  ```
  python src/kg_model.py
  ```
  Trains a heterogeneous GNN for KG embedding. Checkpoints saved to `models/best_kg_model.pt`.

- **Train Fusion Model:**  
  ```
  python src/train_fusion.py
  ```
  Trains a fusion model combining molecular and KG embeddings.

- **Test Fusion Model:**  
  ```
  python src/test_fusion.py
  ```
  Evaluates the fusion model.

## Notebooks

- Notebooks are not included in the repository.  
- To use Jupyter notebooks for exploratory data analysis, visualization, and prototyping, create your own `notebooks/` directory locally and add your `.ipynb` files as needed.

## How to Run

- All scripts are in `src/`. Run them with:
  ```
  python src/<script_name>.py
  ```
- Ensure all data, models, and embeddings are in their respective directories.

## Requirements

- Python 3.8+
- CUDA-enabled GPU recommended for training
- See `requirements.txt` for full package list

## Acknowledgments

- ZINC dataset: [https://zinc.docking.org/](https://zinc.docking.org/)
- PyTorch Geometric: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
- RDKit: [https://www.rdkit.org/](https://www.rdkit.org/)

## Contact

For questions or contributions, please open an issue or pull request.