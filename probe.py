import torch
import torch.nn as nn
import numpy as np
import h5py
import torchvision
from models.encoder import Encoder

DATASET = "so100-data/svla_so100_pickplace.h5"
CKPT = "lewm.pt"
N_SAMPLES = 2000
CAMERA = "pixels_top"
device = ""
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

with h5py.File(DATASET, "r") as f:
    N = f["joint_pos"].shape[0]
    idx = np.sort(np.random.choice(N, size=min(N_SAMPLES, N), replace=False))
    imgs = f[f"observation/{CAMERA}"][idx]
    jpos = f["joint_pos"][idx]
    episode_ids = f["episode_index"][idx]
imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
Y = torch.from_numpy(jpos).float()
Yn = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-6)   # standardize targets
print("probing on", len(imgs), "frames")

# three encoders to compare

# LeWM (my model)
lewm = Encoder(img_size=224, patch=16, in_ch=3, dim=192, depth=12, heads=3).to(device)
lewm.load_state_dict(torch.load(CKPT, map_location=device)["encoder"])

@torch.no_grad()
def recalibrate_bn(enc, X, bs=64):
    enc.train()

    bn = enc.proj_bn
    old_momentum = bn.momentum

    bn.reset_running_stats()
    bn.momentum = None

    for i in range(0, len(X), bs):
        enc(X[i:i + bs].to(device))

    bn.momentum = old_momentum
    enc.eval()

# recalibrate_bn(lewm, imgs)

# Random
rand = Encoder(img_size=224, patch=16, in_ch=3, dim=192, depth=12, heads=3).to(device) 

# ResNet
resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = nn.Identity()
resnet = resnet.to(device)

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

@torch.no_grad()
def embed(enc, X, norm=False, bs=64):
    enc.eval()
    outs = []
    for i in range(0, len(X), bs):
        b = X[i:i+bs]
        if norm: b = (b - mean) / std
        outs.append(enc(b.to(device)).cpu())
    return torch.cat(outs)

# g = torch.Generator().manual_seed(0)               # same split for all three (fair)
# perm = torch.randperm(len(Yn), generator=g)
# cut = int(0.8 * len(Yn))
# tr, te = perm[:cut], perm[cut:]

rng = np.random.default_rng(0)

episodes = np.unique(episode_ids)
rng.shuffle(episodes)

cut = int(0.8 * len(episodes))
train_episodes = episodes[:cut]
test_episodes = episodes[cut:]

tr = torch.from_numpy(
    np.where(np.isin(episode_ids, train_episodes))[0]
).long()

te = torch.from_numpy(
    np.where(np.isin(episode_ids, test_episodes))[0]
).long()

recalibrate_bn(lewm, imgs[tr])

def probe(Z):
    p = nn.Linear(Z.shape[1], Yn.shape[1]); opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        nn.functional.mse_loss(p(Z[tr]), Yn[tr]).backward()
        opt.step()
    with torch.no_grad():
        mse = nn.functional.mse_loss(p(Z[te]), Yn[te]).item()
        base = nn.functional.mse_loss(Yn[te].mean(0, keepdim=True).expand_as(Yn[te]), Yn[te]).item()
    return 1 - mse / base

def mlp_probe(Z):
    net = nn.Sequential(nn.Linear(Z.shape[1], 256), nn.ReLU(), nn.Linear(256, Yn.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(800):
        opt.zero_grad()
        nn.functional.mse_loss(net(Z[tr]), Yn[tr]).backward()
        opt.step()
    with torch.no_grad():
        mse = nn.functional.mse_loss(net(Z[te]), Yn[te]).item()
        base = nn.functional.mse_loss(Yn[te].mean(0, keepdim=True).expand_as(Yn[te]), Yn[te]).item()
    return 1 - (mse / base)

def emb_stats(Z, name):
    stds = Z.std(0)
    sv = torch.linalg.svdvals(Z - Z.mean(0, keepdim=True))
    p = sv / sv.sum(); eff = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    print(f"  {name}: #dims std<0.01 = {(stds<0.01).sum().item()}/{Z.shape[1]} | effective rank {eff:.1f}/{Z.shape[1]}")

for name, Z in [("LeWM", embed(lewm, imgs)),
                ("Random", embed(rand, imgs)),
                ("ResNet18", embed(resnet, imgs, norm=True))]:
    print(f"{name:9s}: linear R^2 {probe(Z):.3f} | MLP R^2 {mlp_probe(Z):.3f}")
    emb_stats(Z, name)

# print(f"LeWM (yours) : R^2 {probe(embed(lewm, imgs)):.3f}")
# print(f"Random encoder : R^2 {probe(embed(rand, imgs)):.3f}")
# print(f"ResNet18 (pretrained) : R^2 {probe(embed(resnet, imgs, norm=True)):.3f}")