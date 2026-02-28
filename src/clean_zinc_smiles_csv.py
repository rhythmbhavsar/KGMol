"""
clean_zinc_smiles_csv.py

Clean the zinc_smiles.csv file:
- Remove rows with missing or invalid SMILES
- Remove rows with missing descriptor values
- Remove duplicate SMILES
- Save cleaned CSV to data/zinc_smiles_clean.csv
"""

import pandas as pd
import os

def clean_zinc_smiles_csv(
input_csv="data/raw/zinc_smiles.csv",
output_csv="data/raw/zinc_smiles_clean.csv"
):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Remove rows with missing or invalid SMILES
    df = df[df["smiles"].notnull() & (df["smiles"] != "")]

    # Remove rows with missing descriptor values (any NaN in descriptor columns)
    descriptor_cols = ["molwt", "logp", "ring_count", "hbd", "hba"]
    df = df.dropna(subset=descriptor_cols)

    # Remove duplicate SMILES (keep first occurrence)
    df = df.drop_duplicates(subset="smiles", keep="first")
    df = df.drop('fail_reason', axis=1)

    # Save cleaned CSV
    df.to_csv(output_csv, index=False)

    # Print summary
    print(f"Cleaned CSV saved to {output_csv}")
    print(f"Total molecules after cleaning: {len(df)}")

if __name__ == "__main__":
    clean_zinc_smiles_csv()