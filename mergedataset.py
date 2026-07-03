import pandas as pd
import glob
import os
import gc
import pyarrow.parquet as pq
from rdkit import Chem

# --- CONFIGURATION ---
MS_FILENAME = 'ms_data_cleaned.parquet'
OUTPUT_FILE = 'MASTER_multimodal_dataset_FULL.parquet'
# ULTRA-SAFE MODE: Process only 2,000 rows at a time
# This keeps memory spikes below ~100MB
BATCH_SIZE = 2000  

def canonicalize_smiles(smiles):
    try:
        if not isinstance(smiles, str): return None
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    return None

def find_file(filename):
    if os.path.exists(filename): return filename
    if os.path.exists(os.path.join('data', filename)): return os.path.join('data', filename)
    files = glob.glob(f"**/{filename}", recursive=True)
    return files[0] if files else None

def main():
    print("========================================================")
    print("   ULTRA-SAFE MEMORY MERGER (Batch Size: 2,000)")
    print("========================================================")

    # --- STEP 1: LOAD MS DATA ---
    print("\n--- STEP 1: Loading Processed MS Data ---")
    ms_path = find_file(MS_FILENAME)
    if not ms_path:
        print(f"❌ CRITICAL ERROR: Could not find '{MS_FILENAME}'")
        return

    # Load MS data (It fits in memory)
    df_ms = pd.read_parquet(ms_path)
    if 'smiles' in df_ms.columns:
        df_ms = df_ms.rename(columns={'smiles': 'canonical_smiles'})
    
    # Create the lookup set
    ms_smiles_set = set(df_ms['canonical_smiles'].dropna())
    print(f"   -> Loaded {len(df_ms)} MS records.")
    print(f"   -> Target molecules to match: {len(ms_smiles_set)}")

    # --- STEP 2: LOCATE IR CHUNKS ---
    ir_files = glob.glob('**/IR_data_chunk*.parquet', recursive=True)
    print(f"\n--- STEP 2: Found {len(ir_files)} IR chunks ---")

    # --- STEP 3: BATCH MATCHING ---
    print("\n--- STEP 3: Matching Data (Slow & Steady Mode) ---")
    all_matches = []
    
    for f_path in ir_files:
        filename = os.path.basename(f_path)
        print(f"\n📂 Processing {filename}...")
        
        try:
            parquet_file = pq.ParquetFile(f_path)
            batch_count = 0
            file_matches = 0
            
            # Using smaller batches to prevent 'realloc' errors
            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
                chunk_df = batch.to_pandas()
                batch_count += 1
                
                col_name = 'smiles' if 'smiles' in chunk_df.columns else 'structure'
                if col_name in chunk_df.columns:
                    chunk_df['canonical_smiles'] = chunk_df[col_name].apply(canonicalize_smiles)
                    matched_batch = chunk_df[chunk_df['canonical_smiles'].isin(ms_smiles_set)]
                    
                    if not matched_batch.empty:
                        all_matches.append(matched_batch)
                        file_matches += len(matched_batch)
                        print(f"   Batch {batch_count}: +{len(matched_batch)} matches (Total: {file_matches})", end='\r')
                
                # AGGRESSIVE MEMORY CLEANUP
                del chunk_df
                if 'matched_batch' in locals(): del matched_batch
                gc.collect() # Force clear RAM every single loop

            print(f"   -> Finished file. Total matches here: {file_matches}")
            
        except Exception as e:
            print(f"   -> [Still Error] Failed on {filename}: {e}")
            gc.collect()

    # --- STEP 4: SAVE ---
    print("\n\n--- STEP 4: Saving Master Dataset ---")
    if all_matches:
        total_ir_df = pd.concat(all_matches)
        master_df = pd.merge(total_ir_df, df_ms, on='canonical_smiles', suffixes=('_IR', '_MS'))
        master_df.to_parquet(OUTPUT_FILE)
        
        print(f"\n✅ SUCCESS! Full Master Dataset Saved: {OUTPUT_FILE}")
        print(f"   Final Dataset Size: {len(master_df)} combined samples")
    else:
        print("⚠️ WARNING: No matches found.")

if __name__ == "__main__":
    main()