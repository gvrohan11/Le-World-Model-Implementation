import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionEncoder(nn.Module):
    def __init__(self, action_dim=6, dim=192):
        super().__init__()
        self.smooth = nn.Conv1d(action_dim, action_dim, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, actions):
        x = actions.float().transpose(1, 2)
        x = self.smooth(x).transpose(1, 2)
        return self.mlp(x)

class _Attention(nn.Module):
    def __init__(self,dim=192, heads=16, dim_head=64, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch, length, _ = x.shape
        qkv = self.to_qkv(self.norm(x))
        qkv = qkv.view(batch, length, 3, self.heads, self.dim_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout if self.training else 0.0)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch, length, self.heads * self.dim_head)
        return self.to_out(attended)

class _ConditionalBlock(nn.Module):
    def __init__(self, dim=192, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = _Attention(dim, heads, dim_head, dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale) + shift

    def forward(self, x, action_embedding):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(action_embedding).chunk(6, dim=-1)
        )
        x = x + gate_a * self.attention(self._modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m * self.mlp(self._modulate(self.norm2(x), shift_m, scale_m))
        return x

class Predictor(nn.Module):
    def __init__(self, dim=192, num_frames=3, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, num_frames, dim))
        self.blocks = nn.ModuleList(
            [
                _ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, action_embedding):
        if z.shape[1] > self.position.shape[1]:
            raise ValueError(
                f"Sequence length {z.shape[1]} exceeds configured "
                f"length {self.position.shape[1]}"
            )
        x = z + self.position[:, : z.shape[1]]
        for block in self.blocks:
            x = block(x, action_embedding)
        return self.norm(x)
    