import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import time
from rdkit import Chem

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_FILE = 'MASTER_multimodal_dataset_FINAL_HUGE.parquet'
MODEL_SAVE_PATH = 'TOXICITY_GNN_BEST.pth'
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
MAX_ATOMS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Must match app.py exactly
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Unknown']

print(f"🚀 INITIALIZING GNN SAFETY MODULE TRAINING on {DEVICE}")

# ==========================================
# 2. MEDICINAL CHEMISTRY RULES (LABELING)
# ==========================================
# We define toxicophores. If a molecule has these, it is labeled Toxic (1).
TOXIC_SMARTS = [
    '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]', # Nitro groups (Explosives/Toxic)
    '[CX3](=[OX1])[F,Cl,Br,I]',               # Acid halides (Highly reactive)
    '[CX2]#[NX1]',                            # Cyanides/Nitriles (Poison)
    '[OX2](-[OX2])',                          # Peroxides (Explosive/Reactive)
    'C(=O)C=C',                               # Michael Acceptors (DNA binders)
    '[#15]'                                   # Phosphorous (Nerve agents/Pesticides)
]
compiled_smarts = [Chem.MolFromSmarts(sm) for sm in TOXIC_SMARTS]

def is_toxic(smiles):
    """Returns 1.0 if toxic, 0.0 if safe"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return 0.0
        for patt in compiled_smarts:
            if mol.HasSubstructMatch(patt):
                return 1.0
        return 0.0
    except:
        return 0.0

# ==========================================
# 3. GRAPH DATASET & CONVERSION
# ==========================================
def smiles_to_graph_arrays(smiles, max_atoms=MAX_ATOMS):
    """Converts SMILES to Graph Nodes and Adjacency Matrix (Numpy)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        
        num = min(mol.GetNumAtoms(), max_atoms)
        x = np.zeros((max_atoms, len(ATOM_LIST)), dtype=np.float32)
        adj = np.zeros((max_atoms, max_atoms), dtype=np.float32)
        
        for i in range(num):
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            idx = ATOM_LIST.index(sym) if sym in ATOM_LIST else 9
            x[i, idx] = 1.0
            
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i < max_atoms and j < max_atoms:
                adj[i, j] = adj[j, i] = 1.0
                
        np.fill_diagonal(adj, 1.0)
        deg = np.clip(np.sum(adj, axis=1), 1, None)
        d_inv = np.power(deg, -0.5)
        adj = (adj * d_inv[:, None]) * d_inv[None, :]
        
        return x, adj
    except:
        return None, None

class ToxicityDataset(Dataset):
    def __init__(self, parquet_file):
        print(f"⏳ Loading Dataset from {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        # Find SMILES column
        cols = df.columns
        smiles_col = next((c for c in cols if 'canon' in c.lower() or 'smi' in c.lower()), None)
        
        # Keep only unique SMILES to speed up training
        self.smiles_list = df[smiles_col].dropna().unique().tolist()
        print(f"✅ Found {len(self.smiles_list)} unique molecules.")
        
        # Pre-calculate Labels to find the Imbalance Weight
        print("⏳ Scanning molecules against Toxicity PAINS filters...")
        self.labels = []
        self.valid_smiles = []
        self.graphs_x = []
        self.graphs_adj = []
        
        toxic_count = 0
        safe_count = 0
        
        for smi in self.smiles_list:
            x, adj = smiles_to_graph_arrays(smi)
            if x is not None:
                label = is_toxic(smi)
                if label == 1.0: toxic_count += 1
                else: safe_count += 1
                
                self.valid_smiles.append(smi)
                self.labels.append(label)
                self.graphs_x.append(x)
                self.graphs_adj.append(adj)
                
        # --- CRITICAL FIX: CALCULATE WEIGHT ---
        # If 90% are Safe and 10% are Toxic, a toxic mistake costs 9x more!
        self.pos_weight = safe_count / max(toxic_count, 1)
        
        print("\n📊 DATASET STATISTICS:")
        print(f"   Safe Molecules  (0): {safe_count}")
        print(f"   Toxic Molecules (1): {toxic_count}")
        print(f"   Calculated Toxic Penalty Weight: {self.pos_weight:.2f}x")

    def __len__(self):
        return len(self.valid_smiles)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.graphs_x[idx]),
            'adj': torch.tensor(self.graphs_adj[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# ==========================================
# 4. GNN MODEL ARCHITECTURE (Matches app.py)
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
    def forward(self, x, adj):
        return self.linear(torch.matmul(adj, x))

class ToxicityGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(len(ATOM_LIST), 64)
        self.gcn2 = GCNLayer(64, 128)
        self.gcn3 = GCNLayer(128, 64)
        # Using Sigmoid at the end to output a 0-1 probability
        self.head = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1), 
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        
    def forward(self, x, adj):
        h = self.relu(self.gcn1(x, adj))
        h = self.relu(self.gcn2(h, adj))
        h = self.relu(self.gcn3(h, adj))
        return self.head(torch.mean(h, dim=1)).squeeze(-1)

# ==========================================
# 5. TRAINING LOOP WITH WEIGHTED LOSS
# ==========================================
def main():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Error: {DATASET_FILE} not found!")
        return

    # 1. Initialize Data
    dataset = ToxicityDataset(DATASET_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Initialize Model
    model = ToxicityGNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Setup Custom Weighted Loss Function
    # We use reduction='none' so we can apply our multiplier to toxic samples manually
    base_loss_fn = nn.BCELoss(reduction='none')
    toxic_weight = dataset.pos_weight

    print("\n🚀 STARTING TRAINING LOOP...")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch in dataloader:
            x = batch['x'].to(DEVICE)
            adj = batch['adj'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(x, adj)
            
            # --- THE MAGIC HAPPENS HERE ---
            # Calculate standard loss
            loss = base_loss_fn(preds, labels)
            
            # Create a multiplier array: 'toxic_weight' for Toxic (1.0), '1.0' for Safe (0.0)
            weight_multiplier = torch.where(labels == 1.0, toxic_weight, 1.0)
            
            # Multiply and average
            weighted_loss = (loss * weight_multiplier).mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            total_loss += weighted_loss.item()
            
            # Tracking basic accuracy (Threshold 0.5)
            binary_preds = (preds > 0.5).float()
            correct_preds += (binary_preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_preds / total_samples) * 100
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}% | Time: {elapsed:.1f}s")
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   💾 New Best Model Saved to {MODEL_SAVE_PATH}")

    print("\n✅ GNN TRAINING COMPLETE!")
    print(f"The model is now properly balanced and saved as {MODEL_SAVE_PATH}.")

if __name__ == "__main__":
    main()