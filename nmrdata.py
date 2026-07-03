import pandas as pd
import glob
import os
import gc
from rdkit import Chem

# --- CONFIGURATION ---
# Your current IR+MS dataset
CURRENT_MASTER = 'MASTER_multimodal_dataset_FULL.parquet' 
# Output file
OUTPUT_FILE = 'MASTER_multimodal_dataset_FINAL_HUGE.parquet'

def canonicalize_smiles(smiles):
    """Standardizes SMILES to ensure we find matches."""
    try:
        if not isinstance(smiles, str): return None
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

def main():
    print("========================================================")
    print("   ADDING MASSIVE EXPERIMENTAL NMR DATA")
    print("========================================================")

    # 1. Load the Master (IR + MS) Dataset
    print(f"\n--- Step 1: Loading Master Dataset ({CURRENT_MASTER}) ---")
    if not os.path.exists(CURRENT_MASTER):
        print("❌ Error: Master dataset not found. Run the merge script first.")
        return
    
    df_master = pd.read_parquet(CURRENT_MASTER)
    # Create a set of SMILES we need to find NMR for
    target_smiles = set(df_master['canonical_smiles'])
    print(f"   -> Looking for NMR data for {len(df_master)} molecules...")

    # 2. Find the Huge NMR File
    print("\n--- Step 2: Locating Huge NMR File ---")
    # Search in the 17296666 folder seen in your screenshots
    nmr_files = glob.glob('**/NMRexp_*.parquet', recursive=True)
    
    if not nmr_files:
        print("❌ Error: Could not find 'NMRexp_*.parquet'.")
        print("   Check if folder '17296666' is unzipped.")
        return
    
    # Pick the largest parquet file found
    nmr_path = max(nmr_files, key=os.path.getsize)
    print(f"   -> Found Huge NMR Database: {nmr_path}")
    print("   -> Loading... (This might take a minute)")
    
    # 3. Load & Filter NMR Data
    # We only keep rows that match our IR+MS molecules to save RAM
    df_nmr = pd.read_parquet(nmr_path)
    
    print(f"   -> Raw NMR Database Size: {len(df_nmr)} records")
    
    # Identify SMILES column (usually 'smiles' or 'canonical_smiles')
    col_name = 'smiles'
    if 'smiles' not in df_nmr.columns:
        # Fallback search
        for c in df_nmr.columns:
            if 'smi' in c.lower(): col_name = c; break
    
    print(f"   -> Standardizing NMR SMILES from column '{col_name}'...")
    df_nmr['canonical_smiles'] = df_nmr[col_name].apply(canonicalize_smiles)
    
    # 4. The Merge
    print("\n--- Step 3: Merging ---")
    # Keep only NMR records that match our Master list
    matched_nmr = df_nmr[df_nmr['canonical_smiles'].isin(target_smiles)]
    
    if matched_nmr.empty:
        print("⚠️ Warning: No matches found in the huge dataset either.")
        print("   This is rare. It implies your IR/MS molecules are very unique.")
    else:
        # Merge them
        df_final = pd.merge(df_master, matched_nmr, on='canonical_smiles', suffixes=('', '_NMR'))
        
        # Save
        df_final.to_parquet(OUTPUT_FILE)
        print(f"\n✅ SUCCESS! Huge 3-Way Dataset Created.")
        print(f"   Saved to: {OUTPUT_FILE}")
        print(f"   FINAL COUNT: {len(df_final)} molecules.")
        print("   (Use this file for training your model!)")

if __name__ == "__main__":
    main()