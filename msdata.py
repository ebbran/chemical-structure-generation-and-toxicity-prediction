import os
from rdkit import Chem
import pandas as pd

def extract_mona_data(sdf_path):
    # Check if the file exists before starting to prevent crashes
    if not os.path.exists(sdf_path):
        print(f"Error: File not found at {sdf_path}")
        return

    print(f"Reading {sdf_path}...")
    # Use ForwardSDMolSupplier for large 2.0 GB files to prevent memory errors
    suppl = Chem.ForwardSDMolSupplier(open(sdf_path, 'rb'))
    ms_records = []

    for i, mol in enumerate(suppl):
        if mol is None: continue
        
        try:
            # 1. Standardize the molecule identity using Canonical SMILES
            smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # 2. Extract peaks from the specific MoNA property tag
            peaks = mol.GetProp('MASS SPECTRAL PEAKS') if mol.HasProp('MASS SPECTRAL PEAKS') else ""
            
            ms_records.append({'smiles': smiles, 'ms_peaks': peaks})
            
            if i % 5000 == 0:
                print(f"Successfully processed {i} molecules...")
        except Exception as e:
            continue

    # Convert the list to a DataFrame and save as Parquet for your project
    df = pd.DataFrame(ms_records)
    df.to_parquet('ms_data_cleaned.parquet', index=False)
    print("Done! Cleaned MS data saved to 'ms_data_cleaned.parquet'.")

# UPDATE: Include the folder name in the path
file_to_process = r'data\MoNA-export-All_Spectra-sdf\MoNA-export-All_Spectra.sdf'
extract_mona_data(file_to_process)