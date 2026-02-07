# module_gnn_model.py（纯 PyTorch GCN 双头）
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, A_hat, X):
        H = torch.matmul(A_hat, X)
        H = self.lin(H)
        H = self.act(H)
        H = self.drop(H)
        return H

class GNNRetVol(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gcn = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.gcn.append(GCNLayer(d, hidden, dropout))
            d = hidden

        self.head_ret = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.head_vol = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )

    def forward(self, A_hat, X):
        H = X
        for layer in self.gcn:
            H = layer(A_hat, H)
        ret = self.head_ret(H).squeeze(-1)
        vol = self.head_vol(H).squeeze(-1)
        return ret, vol
