# HST EN Accurate - Error Networks with ErrorSupervisor & HomeostasisController
from google.colab import drive; drive.mount('/content/drive')
import subprocess, sys, os, torch, torch.nn as nn; subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
import numpy as np; from sklearn.preprocessing import StandardScaler; from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda'); os.makedirs('/content/drive/MyDrive/HST_Training/en_accurate', exist_ok=True)
print("HST EN Accurate - Error Networks")

class ErrorSupervisor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.detector = nn.Linear(d_model, 1); self.corrector = nn.Linear(d_model, d_model); self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        err = torch.sigmoid(self.detector(x))
        correction = self.corrector(x) * err
        return self.norm(x + correction)

class HomeostasisController(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.setpoint = nn.Parameter(torch.randn(d_model)); self.controller = nn.Linear(d_model, d_model); self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        deviation = x - self.setpoint.unsqueeze(0).unsqueeze(0)
        correction = self.controller(deviation) * 0.1
        return self.norm(x - correction)

class ErrorNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.detector = nn.Linear(d_model, 1)
        self.corrector = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Linear(d_model*2, d_model))
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        err = torch.sigmoid(self.detector(x))
        correction = self.corrector(x)
        return self.norm(x + err * correction)

class HSTENAccurate(nn.Module):
    def __init__(self, d_model=32, num_layers=4):
        super().__init__()
        self.input = nn.Linear(1, d_model)
        self.error_nets = nn.ModuleList([ErrorNetwork(d_model) for _ in range(num_layers)])
        self.supervisors = nn.ModuleList([ErrorSupervisor(d_model) for _ in range(num_layers)])
        self.homeostasis = HomeostasisController(d_model)
        self.output = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input(x.unsqueeze(-1) if len(x.shape)==2 else x)
        for err_net, sup in zip(self.error_nets, self.supervisors):
            x = err_net(x); x = sup(x)
        x = self.homeostasis(x)
        return self.output(x)

data = np.array([np.linspace(0, 5, 100) + 2*np.sin(2*np.pi*np.arange(100)/50) + np.random.normal(0, 0.5, 100) for _ in range(1000)])
scaler = StandardScaler(); data = scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(data[:800]).to(device)), batch_size=32, shuffle=True)
model = HSTENAccurate().to(device); opt = torch.optim.Adam(model.parameters(), 1e-3); crit = nn.MSELoss()
for e in range(10):
    for X, in train_loader: opt.zero_grad(); loss = crit(model(X)[:, :-1], X[:, 1:]); loss.backward(); opt.step()
    print(f"Epoch {e+1}: Done")
torch.save(model.state_dict(), '/content/drive/MyDrive/HST_Training/en_accurate/hst_en_accurate.pt')
print("✓ HST EN Accurate saved")
