# HST v4.1 Accurate - Enhanced Unified Modes
from google.colab import drive; drive.mount('/content/drive')
import subprocess, sys, os, torch, torch.nn as nn; subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
import numpy as np; from sklearn.preprocessing import StandardScaler; from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda'); os.makedirs('/content/drive/MyDrive/HST_Training/v4_1_accurate', exist_ok=True)
print("HST v4.1 Accurate - Enhanced Unified")

class EnhancedChunkProcessor(nn.Module):
    def __init__(self, d_model, chunk_size=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8, d_model*4, batch_first=True), num_layers=2)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model, 8, d_model*4, batch_first=True), num_layers=2)
    
    def forward(self, x):
        B, S, D = x.shape
        num_chunks = S // self.chunk_size
        chunks = x[:, :num_chunks*self.chunk_size].view(B*num_chunks, self.chunk_size, D)
        encoded = self.encoder(chunks); decoded = self.decoder(encoded, encoded)
        return decoded.view(B, num_chunks*self.chunk_size, D)

class HSTv41Accurate(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.input = nn.Linear(1, d_model)
        self.processor = EnhancedChunkProcessor(d_model, 16)
        self.output = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input(x.unsqueeze(-1) if len(x.shape)==2 else x)
        x = self.processor(x)
        return self.output(x)

data = np.array([np.linspace(0, 5, 100) + 2*np.sin(2*np.pi*np.arange(100)/50) + np.random.normal(0, 0.5, 100) for _ in range(1000)])
scaler = StandardScaler(); data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(data[:800]).to(device)), batch_size=32, shuffle=True)
model = HSTv41Accurate().to(device); opt = torch.optim.Adam(model.parameters(), 1e-3); crit = nn.MSELoss()
for e in range(10):
    for X, in train_loader: opt.zero_grad(); loss = crit(model(X)[:, :-1].squeeze(-1), X[:, 1:]); loss.backward(); opt.step()
    print(f"Epoch {e+1}: Done")
torch.save(model.state_dict(), '/content/drive/MyDrive/HST_Training/v4_1_accurate/hst_v4_1_accurate.pt')
print("✓ HST v4.1 Accurate saved")
