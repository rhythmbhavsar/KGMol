# Molecular AI Pipeline


### What Is This Project?
This project is like building a smart assistant for scientists working with chemicals. It helps you organize, understand, and predict important things about molecules—like how well they dissolve or how they might behave in a lab. It uses both the usual information about molecules and also how they are connected to each other, making predictions more accurate.

### What Did We Do?

**1. Getting the Data Ready**
- We started with a big list of molecules from a public database (ZINC).
- Cleaned up the data so it’s reliable and easy to use.
- Changed the data into a format that computers can understand, like turning molecules into “maps” showing how their atoms are connected.

**2. Building a Molecule Network (Knowledge Graph)**
- Imagine a social network, but for molecules: each molecule is a “person” and their connections (like similar features or reactions) are “friendships.”
- We built this network so the computer can see not just what a molecule looks like, but also how it relates to others.
- We turned these relationships into numbers so the computer can use them for learning.

**3. Teaching the Computer to Predict**
- We trained two types of computer models:
  - One that looks only at the molecule’s structure (like reading a recipe).
  - Another that looks at both the structure and the network of relationships (like reading the recipe and also asking friends for advice).
- We found that the second model (using both structure and relationships) makes better predictions.

**4. Checking Our Results**
- We used interactive notebooks to explore the data and see how well our models work.
- The model that used both types of information was more accurate, meaning it can help scientists make better decisions.

**5. Tools We Used**
- We used popular science and data tools: Python, PyTorch, RDKit, and others. These are like the “toolbox” for modern scientific computing.

### Why Should Biotech Care?
- **Better Answers:** By combining different types of information, we get more reliable predictions about molecules—useful for drug discovery, making new materials, or understanding chemical reactions.
- **Saves Time:** Scientists can quickly screen lots of molecules and focus on the most promising ones.
- **Easy to Adapt:** The system is organized so you can use it for other types of molecules or research questions.

### How Can You Use It?
- Put your molecule data in the right folders.
- Run the provided scripts (like pressing “go” on a machine).
- Get predictions and insights to help your research.

### The Big Idea
By teaching computers to look at both what molecules are and how they’re connected, we help scientists find answers faster and with more confidence.

---

## Technical Summary (for reference)
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
