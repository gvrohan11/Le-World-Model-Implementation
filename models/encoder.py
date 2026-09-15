import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, img_size, patch, in_ch, dim, depth, heads):
        super().__init__()
        n_patches = (img_size // patch) ** 2 # this 

        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

        self.pos = nn.Parameter(torch.zeros(1, n_patches, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)



    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x += self.pos
        x = self.transformer(x)
        x = self.norm(x)
        return x.mean(dim=1)

if __name__ == "__main__":
    enc = Encoder(img_size=64, patch=8, in_ch=3, dim=256, depth=4, heads=4)
    imgs = torch.randn(16, 3, 64, 64)
    z = enc(imgs)
    print(f"Images in: {tuple(imgs.shape)}")
    print(f"Embeddings: {tuple(z.shape)}")
    print(f"Params: {sum(p.numel() for p in enc.parameters()) // 1000}K")