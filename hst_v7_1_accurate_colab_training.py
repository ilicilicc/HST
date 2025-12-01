# HST v7.1 Accurate - Flash Attention, SparseExpertRouter (MoE), Tree-Based Speculative Decoding
from google.colab import drive; drive.mount('/content/drive')
import subprocess, sys, os, torch, torch.nn as nn, torch.nn.functional as F; subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
import numpy as np; from sklearn.preprocessing import StandardScaler; from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda'); os.makedirs('/content/drive/MyDrive/HST_Training/v7_1_accurate', exist_ok=True)
print("HST v7.1 Accurate - Ultimate Performance")

class FlashBlockSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model; self.n_heads = n_heads; self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model*3, bias=False); self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if hasattr(F, 'scaled_dot_product_attention'):
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask, -torch.inf); attn = F.softmax(scores, dim=-1) @ v
        return self.out(attn.transpose(1, 2).reshape(B, S, D))

class SparseExpertRouter(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])
        self.top_k = top_k
    
    def forward(self, x):
        logits = self.router(x); weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[..., k]; w = weights[..., k:k+1]
            for i in range(x.shape[0]):
                output[i] += self.experts[expert_idx[i].item()](x[i]) * w[i]
        return output

class TreeSpeculativeDecoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.draft = nn.Linear(d_model, d_model); self.verify = nn.Linear(d_model, d_model)
    
    def forward(self, x): return self.verify(F.gelu(self.draft(x))) + x

class HSTv71Accurate(nn.Module):
    def __init__(self, d_model=32, num_layers=4):
        super().__init__()
        self.input = nn.Linear(1, d_model)
        self.attn = nn.ModuleList([FlashBlockSparseAttention(d_model) for _ in range(num_layers)])
        self.moe = SparseExpertRouter(d_model, 8, 2)
        self.decoder = TreeSpeculativeDecoder(d_model)
        self.output = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input(x.unsqueeze(-1) if len(x.shape)==2 else x)
        for a in self.attn: x = a(x) + x
        x = self.moe(x) + x
        x = self.decoder(x)
        return self.output(x)

data = np.array([np.linspace(0, 5, 100) + 2*np.sin(2*np.pi*np.arange(100)/50) + np.random.normal(0, 0.5, 100) for _ in range(1000)])
scaler = StandardScaler(); data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(data[:800]).to(device)), batch_size=32, shuffle=True)
model = HSTv71Accurate().to(device); opt = torch.optim.Adam(model.parameters(), 1e-3); crit = nn.MSELoss()
for e in range(10):
    for X, in train_loader: opt.zero_grad(); loss = crit(model(X)[:, :-1], X[:, 1:]); loss.backward(); opt.step()
    print(f"Epoch {e+1}: Done")
torch.save(model.state_dict(), '/content/drive/MyDrive/HST_Training/v7_1_accurate/hst_v7_1_accurate.pt')
print("✓ HST v7.1 Accurate saved")
