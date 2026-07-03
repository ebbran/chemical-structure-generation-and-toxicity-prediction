import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import random
from collections import deque
import os
import copy

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_FILE = 'MASTER_multimodal_dataset_FINAL_HUGE.parquet'
PRETRAINED_MODEL = "ULTIMATE_MODEL_BEST.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# RL Hyperparameters
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
RL_EPOCHS = 10

print(f"🚀 LAUNCHING PHASE 2: RL FINE-TUNING on {DEVICE}")

# ==========================================
# 2. SHARED CLASSES (COPIED FROM PHASE 1)
# ==========================================
class SMILESTokenizer:
    def __init__(self):
        chars = ["<pad>", "<s>", "</s>", "C", "c", "N", "n", "O", "o", "S", "s", 
                 "F", "P", "I", "Cl", "Br", "B", "b", "Si", "si", 
                 "=", "#", "(", ")", "[", "]", 
                 "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                 "+", "-", "H", "@", "/", "\\", ".", "%"]
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, smiles_string):
        if not isinstance(smiles_string, str): return [self.char_to_idx["<s>"], self.char_to_idx["</s>"]]
        s = smiles_string.replace("Cl", "L").replace("Br", "R").replace("Si", "A").replace("si", "a")
        tokens = [self.char_to_idx["<s>"]]
        for char in s:
            if char == "L": char = "Cl"
            if char == "R": char = "Br"
            if char == "A": char = "Si"
            if char == "a": char = "si"
            if char in self.char_to_idx: tokens.append(self.char_to_idx[char])
        tokens.append(self.char_to_idx["</s>"])
        return tokens

    def decode(self, tokens):
        s = ""
        for t in tokens:
            if t == self.char_to_idx["</s>"]: break
            if t in [self.char_to_idx["<s>"], self.char_to_idx["<pad>"]]: continue
            val = t.item() if torch.is_tensor(t) else t
            s += self.idx_to_char.get(val, '')
        return s

class MultimodalDataset(Dataset):
    def __init__(self, parquet_file):
        print(f"   -> Loading Data from {parquet_file}...")
        self.data = pd.read_parquet(parquet_file)
        cols = self.data.columns
        self.ir_col = next((c for c in cols if 'ir' in c.lower() or 'spectrum' in c.lower()), 'ir_spectra')
        self.ms_col = next((c for c in cols if 'ms' in c.lower() or 'peak' in c.lower()), 'ms_peaks')
        self.nmr_col = next((c for c in cols if 'nmr_proc' in c.lower() or 'processed' in c.lower()), 'NMR_processed')
        self.nmr_type_col = next((c for c in cols if 'nmr_type' in c.lower()), 'NMR_type')
        self.smiles_col = next((c for c in cols if 'canon' in c.lower() or 'smi' in c.lower()), 'canonical_smiles')

    def __len__(self): return len(self.data)

    def parse_ms(self, val):
        vec = np.zeros(1000, dtype=np.float32)
        if isinstance(val, str):
            try:
                tokens = val.replace('\n', ' ').split(' ')
                tokens = [t for t in tokens if t.strip()]
                for i in range(0, len(tokens) - 1, 2):
                    idx = int(float(tokens[i]))
                    if 0 <= idx < 1000: vec[idx] = max(vec[idx], float(tokens[i+1]))
            except: pass
        if vec.max() > 0: vec = vec / vec.max()
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
        if isinstance(nmr_list_str, str):
            try:
                peaks = ast.literal_eval(nmr_list_str)
                is_proton = '1H' in str(nmr_type)
                is_carbon = '13C' in str(nmr_type)
                for peak in peaks:
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
    def __init__(self, base_filters=32, layers=5):
        super(DeepSpectralEncoder, self).__init__()
        self.entry = nn.Sequential(nn.Conv1d(1, base_filters, 11, 2, 5), nn.BatchNorm1d(base_filters), nn.ReLU(), nn.MaxPool1d(2))
        self.blocks = nn.ModuleList([ResidualBlock1D(base_filters) for _ in range(layers)])
        self.mid = nn.Conv1d(base_filters, base_filters*2, 3, 2, 1)
        self.blocks2 = nn.ModuleList([ResidualBlock1D(base_filters*2) for _ in range(layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.entry(x)
        for b in self.blocks: x = b(x)
        x = self.mid(x)
        for b in self.blocks2: x = b(x)
        return self.pool(x).squeeze(-1)

class UltimateHybridAgent(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super(UltimateHybridAgent, self).__init__()
        self.ir_encoder = DeepSpectralEncoder(32, 5)
        self.nmr_encoder = DeepSpectralEncoder(32, 5)
        self.ms_encoder = nn.Sequential(nn.Linear(1000, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 128))
        self.spectral_proj = nn.Linear(256, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 200, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ir, ms, nmr, smiles_seq, tgt_mask=None):
        ir_f = self.ir_encoder(ir.unsqueeze(1))
        nmr_f = self.nmr_encoder(nmr.unsqueeze(1))
        ms_f = self.ms_encoder(ms)
        memory = self.spectral_proj(torch.cat([ir_f, ms_f, nmr_f], dim=1)).unsqueeze(1)
        seq_len = smiles_seq.size(1)
        tgt = self.embedding(smiles_seq) + self.pos_encoder[:, :seq_len, :]
        out = self.transformer(tgt, memory, tgt_mask=tgt_mask)
        return self.head(out)

# ==========================================
# 3. RL ENVIRONMENT
# ==========================================
class ChemEnv:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.current_idx = random.randint(0, len(self.dataset)-1)
        self.sample = self.dataset[self.current_idx]
        self.state_smiles = [self.tokenizer.char_to_idx["<s>"]]
        self.target_smiles = self.tokenizer.encode(self.sample['smiles'])
        return self._get_state()

    def _get_state(self):
        return {
            'ir': self.sample['ir'].to(DEVICE).unsqueeze(0),
            'ms': self.sample['ms'].to(DEVICE).unsqueeze(0),
            'nmr': self.sample['nmr'].to(DEVICE).unsqueeze(0),
            'smiles': torch.tensor([self.state_smiles], dtype=torch.long).to(DEVICE)
        }

    def step(self, action_idx):
        self.state_smiles.append(action_idx)
        reward = 0
        done = False
        
        # Determine Success
        if len(self.state_smiles) >= 150:
            done = True
            reward = -1 # Stuck penalty
        elif action_idx == self.tokenizer.char_to_idx["</s>"]:
            done = True
            generated = self.tokenizer.decode(self.state_smiles)
            target = self.sample['smiles']
            if generated == target: reward = 10 # Jackpot
            else: reward = 1.0 / (abs(len(generated) - len(target)) + 1)
        else:
            # Step reward
            pos = len(self.state_smiles) - 1
            if pos < len(self.target_smiles) and action_idx == self.target_smiles[pos]:
                reward = 0.2
            else:
                reward = -0.1

        return self._get_state(), reward, done

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, *args): self.buffer.append(args)
    def sample(self, batch_size): return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

# ==========================================
# 4. MAIN RL LOOP
# ==========================================
def main():
    tokenizer = SMILESTokenizer()
    dataset = MultimodalDataset(DATASET_FILE)
    env = ChemEnv(dataset, tokenizer)
    buffer = ReplayBuffer(MEMORY_SIZE)
    
    # Load Models
    print(f"   Loading Pre-trained Weights: {PRETRAINED_MODEL}")
    if not os.path.exists(PRETRAINED_MODEL):
        print("❌ CRITICAL ERROR: Could not find 'ULTIMATE_MODEL_BEST.pth'")
        return

    policy_net = UltimateHybridAgent(tokenizer.vocab_size).to(DEVICE)
    policy_net.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=DEVICE))
    
    target_net = UltimateHybridAgent(tokenizer.vocab_size).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00001) # Low LR for fine-tuning
    epsilon = EPSILON_START

    print("🚀 Starting RL Loop...")
    for epoch in range(RL_EPOCHS):
        state = env.reset()
        total_reward = 0
        
        for step in range(500): # Steps per epoch
            # 1. Action
            if random.random() < epsilon:
                action = random.randint(0, tokenizer.vocab_size - 1)
            else:
                with torch.no_grad():
                    q = policy_net(state['ir'], state['ms'], state['nmr'], state['smiles'])
                    action = q[0, -1, :].argmax().item()

            # 2. Step
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done: state = env.reset()

            # 3. Train
            if len(buffer) > BATCH_SIZE:
                transitions = buffer.sample(BATCH_SIZE)
                # Simplified batch update for stability
                batch = transitions[0] 
                s, a, r, ns, d = batch
                
                q_curr = policy_net(s['ir'], s['ms'], s['nmr'], s['smiles'])[0, -1, a]
                
                with torch.no_grad():
                    if d: target = r
                    else:
                        q_next = target_net(ns['ir'], ns['ms'], ns['nmr'], ns['smiles'])
                        target = r + GAMMA * q_next[0, -1, :].max().item()
                
                # --- THE FIX: Force target to be float ---
                target_tensor = torch.tensor(target, dtype=torch.float32).to(DEVICE)
                loss = nn.MSELoss()(q_curr, target_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if epoch % 2 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"   🔄 Target Updated. Saving Checkpoint...")
            torch.save(policy_net.state_dict(), "ULTIMATE_MODEL_RL_TUNED.pth")

        print(f"Epoch {epoch+1}/{RL_EPOCHS} | Reward: {total_reward:.2f} | Eps: {epsilon:.3f}")

    print("🏆 RL FINETUNING COMPLETE!")

if __name__ == "__main__":
    main()