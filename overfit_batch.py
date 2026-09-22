import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SO100Pairs
from models.encoder import Encoder
from models.predictor import Predictor
from losses.sigreg import sigreg_loss

'''
Diagnostic script
Loads one fixed batch of 62 one-step transitions
Repeatedly trains on that same batch
Ideally, we want the prediction loss to be close to 0

Core operations: concatenate curr and next frame
encode that concatentation
calculate the new curr and new next from that
predict new next using Predictor
calculate predicted loss by comparing z_hat with z_next
'''


DATASET = "so100-data/svla_so100_pickplace.h5"
BATCH = 62
STEPS = 2000 #1000
device = ""
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

ds = SO100Pairs(DATASET, gap=1)
loader = DataLoader(
    ds,
    batch_size=BATCH,
    shuffle=False,
    drop_last=True,
    num_workers=0
)

frame, next_frame, action = next(iter(loader))

frame = frame.to(device)
next_frame = next_frame.to(device)
action = action.to(device)

encoder = Encoder(img_size=224, patch=16, in_ch=3, dim=192, depth=12, heads=3).to(device)
predictor = Predictor(dim=192, action_dim=6, hidden=512).to(device)

# encoder.train()

# with torch.no_grad():
#     frames = torch.cat([frame, next_frame], dim=0)
#     z_all = encoder(frames)
#     z, z_next = z_all.chunk(2, dim=0)

# for parameter in encoder.parameters():
#     parameter.requires_grad_(False)

opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-4) # WAS 1e-3

opt = torch.optim.Adam([
    {
        "params": encoder.parameters(),
        "lr": 1e-5
    },
    {
        "params": predictor.parameters(),
        "lr": 1e-3
    }
])

# opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

for step in range(1, STEPS + 1):
    frames = torch.cat([frame, next_frame], dim=0)
    z_all = encoder(frames)
    z, z_next = z_all.chunk(2, dim=0)
    z_hat = predictor(z, action)
    pred_loss = nn.functional.mse_loss(z_hat, z_next.detach())
    loss = pred_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step == 1 or step % 200 == 0:
        with torch.no_grad():
            z_std = z.std(dim=0)
        print(
            f"step {step:4d} | "
            f"pred {pred_loss.item():.6f} "
            f"std {z_std.mean().item():.4f}"
        )

# for step in range(1, 2001):
#     z_hat = predictor(z, action)
#     pred_loss = nn.functional.mse_loss(z_hat, z_next)
#     opt.zero_grad(set_to_none=True)
#     pred_loss.backward()
#     opt.step()
#     if step == 1 or step % 200 == 0:
#         print(
#             f"step {step:4d} | "
#             f"pred {pred_loss.item():.6f}"
#         )

# for step in range(1, STEPS + 1):
#     frames = torch.cat([frame, next_frame], dim=0)
#     z_all = encoder(frames)
#     z, z_next = z_all.chunk(2, dim=0)
#     z_hat = predictor(z, action)
#     pred_loss = nn.functional.mse_loss(z_hat, z_next)
#     reg_loss = 0.5 * (sigreg_loss(z) + sigreg_loss(z_next))
#     loss = pred_loss + (0.1 * reg_loss)
#     opt.zero_grad(set_to_none=True)
#     loss.backward()
#     opt.step()
#     if step == 1 or step % 100 == 0:
#         with torch.no_grad():
#             std = z.std(dim=0)
#         print(
#             f"step {step:4d} | "
#             f"total {loss.item():.4f} | "
#             f"pred {pred_loss.item():.6f} | "
#             f"sigreg {reg_loss.item():.4f} | "
#             f"std {std.mean().item():.4f}"
#         )