import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import math
import os
import time

# ==========================================
# 1. ADVANCED CONFIGURATION
# ==========================================
DATASET_FILE = 'MASTER_multimodal_dataset_FINAL_HUGE.parquet'
BATCH_SIZE = 32          # Standard batch size for stability
EPOCHS = 30              # Increased for "Ultimate" convergence
LR_MAX = 0.0003          # Peak learning rate for OneCycle scheduler
D_MODEL = 512            # Embedding dimension (Power of 2)
N_HEADS = 8              # Attention heads
N_LAYERS = 6             # Transformer depth (Deep!)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🚀 LAUNCHING ULTIMATE HYBRID AI SYSTEM on {DEVICE}")
print(f"   Target: {DATASET_FILE}")
print(f"   Architecture: Deep ResNet-1D + {N_LAYERS}-Layer Transformer")

# ==========================================
# 2. ROBUST VOCABULARY
# ==========================================
class SMILESTokenizer:
    def __init__(self):
        # Comprehensive character set for organic chemistry
        chars = ["<pad>", "<s>", "</s>", "C", "c", "N", "n", "O", "o", "S", "s", 
                 "F", "P", "I", "Cl", "Br", "B", "b", "Si", "si", 
                 "=", "#", "(", ")", "[", "]", 
                 "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                 "+", "-", "H", "@", "/", "\\", ".", "%"]
        
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, smiles_string):
        """Robust encoding handling multi-char atoms"""
        if not isinstance(smiles_string, str): return [self.char_to_idx["<s>"], self.char_to_idx["</s>"]]
        
        # Canonical replacements to single tokens
        s = smiles_string.replace("Cl", "L").replace("Br", "R").replace("Si", "A").replace("si", "a")
        tokens = [self.char_to_idx["<s>"]]
        
        for char in s:
            if char == "L": char = "Cl"
            if char == "R": char = "Br"
            if char == "A": char = "Si"
            if char == "a": char = "si"
            
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
        
        tokens.append(self.char_to_idx["</s>"])
        return tokens

    def decode(self, tokens):
        s = ""
        for t in tokens:
            if t == self.char_to_idx["</s>"]: break
            if t in [self.char_to_idx["<s>"], self.char_to_idx["<pad>"]]: continue
            s += self.idx_to_char.get(t.item() if torch.is_tensor(t) else t, '')
        return s

# ==========================================
# 3. SMART DATASET LOADER
# ==========================================
class MultimodalDataset(Dataset):
    def __init__(self, parquet_file):
        print(f"   -> Loading Data from {parquet_file}...")
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"CRITICAL: Could not find {parquet_file}")
            
        self.data = pd.read_parquet(parquet_file)
        print(f"   -> Loaded {len(self.data)} samples.")
        
        # Auto-detect column names with fallback logic
        cols = self.data.columns
        self.ir_col = next((c for c in cols if 'ir' in c.lower() or 'spectrum' in c.lower()), 'ir_spectra')
        self.ms_col = next((c for c in cols if 'ms' in c.lower() or 'peak' in c.lower()), 'ms_peaks')
        self.nmr_col = next((c for c in cols if 'nmr_proc' in c.lower() or 'processed' in c.lower()), 'NMR_processed')
        self.nmr_type_col = next((c for c in cols if 'nmr_type' in c.lower()), 'NMR_type')
        self.smiles_col = next((c for c in cols if 'canon' in c.lower() or 'smi' in c.lower()), 'canonical_smiles')

    def __len__(self):
        return len(self.data)

    def parse_ms(self, val):
        vec = np.zeros(1000, dtype=np.float32)
        if not isinstance(val, str): return vec
        try:
            tokens = val.replace('\n', ' ').split(' ')
            tokens = [t for t in tokens if t.strip()]
            for i in range(0, len(tokens) - 1, 2):
                m = float(tokens[i])
                inten = float(tokens[i+1])
                idx = int(m)
                if 0 <= idx < 1000:
                    vec[idx] = max(vec[idx], inten)
        except: pass
        if vec.max() > 0: vec = vec / vec.max() # Normalize
        return vec

    def parse_ir(self, val):
        vec = np.zeros(3000, dtype=np.float32)
        if hasattr(val, '__iter__') and not isinstance(val, str):
            arr = np.array(val, dtype=np.float32)
            length = min(len(arr), 3000)
            vec[:length] = arr[:length]
        return vec

    def parse_nmr(self, nmr_list_str, nmr_type):
        vec = np.zeros(2000, dtype=np.float32)
        if not isinstance(nmr_list_str, str): return vec
        try:
            peaks = ast.literal_eval(nmr_list_str)
            is_proton = '1H' in str(nmr_type)
            is_carbon = '13C' in str(nmr_type)
            for peak in peaks:
                # Expecting tuple: (shape, coupling, nuc, start, end, ...)
                if len(peak) >= 5:
                    ppm = (float(peak[3]) + float(peak[4])) / 2.0
                    if is_proton:
                        idx = int((ppm / 12.0) * 1000)
                        if 0 <= idx < 1000: vec[idx] = 1.0
                    elif is_carbon:
                        idx = int((ppm / 220.0) * 1000)
                        if 0 <= idx < 1000: vec[1000 + idx] = 1.0
        except: pass
        return vec

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'ir': torch.tensor(self.parse_ir(row[self.ir_col])),
            'ms': torch.tensor(self.parse_ms(row[self.ms_col])),
            'nmr': torch.tensor(self.parse_nmr(row[self.nmr_col], row[self.nmr_type_col])),
            'smiles': row[self.smiles_col]
        }

# ==========================================
# 4. DEEP RESNET + TRANSFORMER ARCHITECTURE
# ==========================================
class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super(ResidualBlock1D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 1, padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + x)

class DeepSpectralEncoder(nn.Module):
    """
    Very Deep ResNet Encoder (5 Layers) to catch tiny spectral dips.
    """
    def __init__(self, base_filters=32, layers=5):
        super(DeepSpectralEncoder, self).__init__()
        # Initial Receptive Field
        self.entry = nn.Sequential(
            nn.Conv1d(1, base_filters, 11, 2, 5), 
            nn.BatchNorm1d(base_filters), 
            nn.ReLU(), 
            nn.MaxPool1d(2)
        )
        
        # Deep Residual Stacks
        self.blocks = nn.ModuleList([ResidualBlock1D(base_filters) for _ in range(layers)])
        
        # Downsample & Go Deeper
        self.mid = nn.Conv1d(base_filters, base_filters*2, 3, 2, 1)
        self.blocks2 = nn.ModuleList([ResidualBlock1D(base_filters*2) for _ in range(layers)])
        
        # Global Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.entry(x)
        for b in self.blocks: x = b(x)
        x = self.mid(x)
        for b in self.blocks2: x = b(x)
        return self.pool(x).squeeze(-1)

class UltimateHybridAgent(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL):
        super(UltimateHybridAgent, self).__init__()
        
        # --- THE EYES (Sensors) ---
        self.ir_encoder = DeepSpectralEncoder(32, 5)   # Output: 64 dim
        self.nmr_encoder = DeepSpectralEncoder(32, 5)  # Output: 64 dim
        self.ms_encoder = nn.Sequential(               # Output: 128 dim
            nn.Linear(1000, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # Fusion Projection: 64(IR) + 128(MS) + 64(NMR) = 256 -> 512
        self.spectral_proj = nn.Linear(256, d_model)
        
        # --- THE BRAIN (Transformer) ---
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 200, d_model))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=N_HEADS, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=N_LAYERS)
        
        # --- THE DECISION HEAD ---
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ir, ms, nmr, smiles_seq, tgt_mask=None):
        # 1. Sense Spectra
        ir_f = self.ir_encoder(ir.unsqueeze(1))
        nmr_f = self.nmr_encoder(nmr.unsqueeze(1))
        ms_f = self.ms_encoder(ms)
        
        # 2. Create Memory Context
        # [Batch, 1, d_model] - Treating spectra as a single powerful context token
        memory = self.spectral_proj(torch.cat([ir_f, ms_f, nmr_f], dim=1)).unsqueeze(1)
        
        # 3. Process Sequence
        seq_len = smiles_seq.size(1)
        tgt = self.embedding(smiles_seq) + self.pos_encoder[:, :seq_len, :]
        
        # 4. Decode (Attend to Memory)
        out = self.transformer(tgt, memory, tgt_mask=tgt_mask)
        
        return self.head(out)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def create_causal_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1).bool().to(DEVICE)

def main():
    # 1. Setup Data
    tokenizer = SMILESTokenizer()
    dataset = MultimodalDataset(DATASET_FILE)
    
    def collate_fn(batch):
        ir = torch.stack([b['ir'] for b in batch])
        ms = torch.stack([b['ms'] for b in batch])
        nmr = torch.stack([b['nmr'] for b in batch])
        
        tokens = [torch.tensor(tokenizer.encode(b['smiles'])) for b in batch]
        max_len = max(len(t) for t in tokens)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, t in enumerate(tokens):
            padded[i, :len(t)] = t
        return ir, ms, nmr, padded

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 2. Setup Model
    model = UltimateHybridAgent(vocab_size=tokenizer.vocab_size).to(DEVICE)
    
    # OneCycleLR Scheduler for Super-Convergence
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, steps_per_epoch=len(dataloader), epochs=EPOCHS
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <pad>
    
    print(f"\n✅ SYSTEM READY: {sum(p.numel() for p in model.parameters()):,} Parameters")
    print(f"✅ Training for {EPOCHS} Epochs...")

    best_loss = float('inf')

    # 3. Train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for i, (ir, ms, nmr, smiles) in enumerate(dataloader):
            ir, ms, nmr, smiles = ir.to(DEVICE), ms.to(DEVICE), nmr.to(DEVICE), smiles.to(DEVICE)
            
            # Input: <s> C C ...
            decoder_input = smiles[:, :-1]
            # Target: C C ... </s>
            target_output = smiles[:, 1:]
            
            mask = create_causal_mask(decoder_input.size(1))
            
            optimizer.zero_grad()
            preds = model(ir, ms, nmr, decoder_input, mask)
            
            # Flatten for Loss: [Batch*Seq, Vocab] vs [Batch*Seq]
            loss = criterion(preds.reshape(-1, tokenizer.vocab_size), target_output.reshape(-1))
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"   Epoch {epoch+1} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"🎉 Epoch {epoch+1} Complete ({elapsed:.1f}s) | Avg Loss: {avg_loss:.4f}")
        
        # Save "Ultimate" Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "ULTIMATE_MODEL_BEST.pth")
            print("   💾 New Best Model Saved!")

        # Always save latest checkpoint
        torch.save(model.state_dict(), "ULTIMATE_MODEL_LATEST.pth")

    print("\n🏆 TRAINING COMPLETE.")
    print("   The ultimate model is saved as 'ULTIMATE_MODEL_BEST.pth'")

if __name__ == "__main__":
    main()