import torch
import torch.nn as nn

'''
Predictor: this takes the current frame's embedding (from encoder) + action the robot took (from policy) and predicts the next frame's embedding
'''

class Predictor(nn.Module):
    def __init__(self, dim=256, action_dim=7, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        return self.net(za)

if __name__ == "__main__":
    pred = Predictor(dim=192, action_dim=6, hidden=512)
    z = torch.randn(16, 192)
    a = torch.randn(16, 6)
    z_next_hat = pred(z, a)
    print(f"Current Embedding: {tuple(z.shape)}")
    print(f"Action: {tuple(a.shape)}")
    print(f"Predicted Next: {tuple(z_next_hat.shape)}")
    print(f"Params: {sum(p.numel() for p in pred.parameters()) // 1000}K")

