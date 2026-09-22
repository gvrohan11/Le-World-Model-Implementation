import torch
import torch.nn as nn

'''
Building the encoder: takes a camera image as input and converts it to a short list of numbers capturing what's in the scene
This file will produce embeddings for sigreg
Everything from model will read from these embeddings
This encoder is untrained. The training (sigreg + predictor) turns these embeddings (number list) into one vector with meaning
'''

class Encoder(nn.Module):
    def __init__(self, img_size=224, patch=16, in_ch=3, dim=192, depth=12, heads=3, out_dim=192):
        super().__init__()
        n_patches = (img_size // patch) ** 2 # this 

        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))

        layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=dim * 4, 
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, out_dim)
        self.proj_bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        x += self.pos
        x = self.transformer(x)
        x = self.norm(x)

        z = self.proj(x[:, 0])

        return self.proj_bn(z)
        # return x.mean(dim=1)

if __name__ == "__main__":
    enc = Encoder(img_size=224, patch=16, in_ch=3, dim=192, depth=12, heads=3)
    imgs = torch.randn(16, 3, 224, 224)
    z = enc(imgs)
    print(f"Images in: {tuple(imgs.shape)}")
    print(f"Embeddings: {tuple(z.shape)}")
    print(f"Params: {sum(p.numel() for p in enc.parameters()) // 1000}K")