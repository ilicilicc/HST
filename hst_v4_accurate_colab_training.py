# HST v4 Accurate - ChunkEncoder/Decoder Architecture
from google.colab import drive; drive.mount('/content/drive')
import subprocess, sys, os, torch, torch.nn as nn; subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
import numpy as np; from sklearn.preprocessing import StandardScaler; from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda'); os.makedirs('/content/drive/MyDrive/HST_Training/v4_accurate', exist_ok=True)
print("HST v4 Accurate - Chunk Encoding/Decoding")

class ChunkEncoder(nn.Module):
    def __init__(self, d_model, chunk_size=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.local_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 8, d_model*4, batch_first=True),
            num_layers=2
        )
        self.chunk_pooling = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
    
    def forward(self, tokens):
        B, total_tokens, D = tokens.shape
        num_chunks = total_tokens // self.chunk_size
        chunks = tokens[:, :num_chunks*self.chunk_size].view(B*num_chunks, self.chunk_size, D)
        encoded = self.local_encoder(chunks)
        pooled = encoded.mean(dim=1)
        compressed = self.chunk_pooling(pooled)
        return compressed.view(B, num_chunks, D)

class ChunkDecoder(nn.Module):
    def __init__(self, d_model, chunk_size=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_expander = nn.Linear(d_model, d_model*chunk_size)
        self.local_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, 8, d_model*4, batch_first=True),
            num_layers=2
        )
        self.lm_head = nn.Linear(d_model, 1)
    
    def forward(self, chunk_embeddings):
        B, num_chunks, D = chunk_embeddings.shape
        expanded = self.chunk_expander(chunk_embeddings).view(B*num_chunks, self.chunk_size, D)
        refined = self.local_decoder(expanded, expanded)
        return self.lm_head(refined.view(B, num_chunks*self.chunk_size, D))

class HSTv4Accurate(nn.Module):
    def __init__(self, d_model=32, chunk_size=16):
        super().__init__()
        self.input = nn.Linear(1, d_model)
        self.chunk_encoder = ChunkEncoder(d_model, chunk_size)
        self.chunk_decoder = ChunkDecoder(d_model, chunk_size)
    
    def forward(self, x):
        x = self.input(x.unsqueeze(-1) if len(x.shape)==2 else x)
        chunk_embs = self.chunk_encoder(x)
        return self.chunk_decoder(chunk_embs)

data = np.array([np.linspace(0, 5, 100) + 2*np.sin(2*np.pi*np.arange(100)/50) + np.random.normal(0, 0.5, 100) for _ in range(1000)])
scaler = StandardScaler(); data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(data[:800]).to(device)), batch_size=32, shuffle=True)
model = HSTv4Accurate().to(device); opt = torch.optim.Adam(model.parameters(), 1e-3); crit = nn.MSELoss()
for e in range(10):
    for X, in train_loader: opt.zero_grad(); loss = crit(model(X)[:, :-1].squeeze(-1), X[:, 1:]); loss.backward(); opt.step()
    print(f"Epoch {e+1}: Done")
torch.save(model.state_dict(), '/content/drive/MyDrive/HST_Training/v4_accurate/hst_v4_accurate.pt')
print("✓ HST v4 Accurate saved")
