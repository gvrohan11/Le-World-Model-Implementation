import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.encoder import Encoder
from models.predictor import Predictor
from losses.sigreg import sigreg_loss
from dataset import SO100Pairs

import time
from datetime import datetime

'''
Training process:
1. Encode current frame
2. Encode the real next frame (real answer)
3. Predictor guesses the next embedding from current frame + action (predicted answer)
4. Calculate Prediction loss: real - predicted
5. Calculate how un-spread the embeddings are (sigreg)
6. Nudge the encoder and predictor to shrink that loss and un-spreadedness
7. Repeat MAX_STEPS times
'''

DATASET = "so100-data/svla_so100_pickplace.h5"
BATCH = 62
LR = 1e-4
MAX_STEPS = 15000 # 50000
SIGREG_W = 25.0 # 1.0
LOG_EVERY = 50
CKPT_EVERY = 1000
CKPT_PATH = "lewm.pt"

def main():
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    encoder = Encoder(img_size=224, patch=16, in_ch=3, dim=192, depth=12, heads=3).to(device)
    predictor = Predictor(dim=192, action_dim=6, hidden=512).to(device)

    ds = SO100Pairs(DATASET)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=(device == "cuda"), drop_last=True)
    print(f"Training Pairs: {len(ds)}")
    opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=LR)

    start_time = time.time()
    print(f"Training started at {datetime.now().strftime('%H:%M:%S')}")

    step = 0
    while step < MAX_STEPS:
        for frame, next_frame, action in loader:
            frame = frame.to(device, non_blocking=True)
            next_frame = next_frame.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)
            z = encoder(frame)
            z_next = encoder(next_frame)
            z_hat = predictor(z, action) # predict next embedding from z

            pred_loss = nn.functional.mse_loss(z_hat, z_next)
            reg_loss = sigreg_loss(z)
            loss = pred_loss + (SIGREG_W * reg_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            if step % LOG_EVERY == 0:
                time_now = datetime.now().strftime('%H:%M:%S')
                print(f"[{time_now}] step {step:6d} | total {loss.item():.4f} | pred {pred_loss.item():.6f} | sigreg {reg_loss.item():.4f}")
            if step % CKPT_EVERY == 0:
                time_now = datetime.now().strftime('%H:%M:%S')
                torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                            "optimizer": opt.state_dict(), "step": step}, CKPT_PATH)
                print(f"  [{time_now}] [checkpoint saved at step {step}]")
            if step >= MAX_STEPS:
                break

    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                "optimizer": opt.state_dict(), "step": step}, CKPT_PATH)
    print("training complete — final checkpoint saved to", CKPT_PATH)

    end_time = time.time()
    print(f"Training ended at {datetime.now().strftime('%H:%M:%S')}")

    total_seconds = end_time - start_time
    print(f"Total training time: {total_seconds:.1f} seconds  ({total_seconds/60:.1f} minutes)")

if __name__ == "__main__":
    main()