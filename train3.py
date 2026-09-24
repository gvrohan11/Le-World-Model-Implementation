import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SO100Sequences
from losses.sigreg2 import sigreg_loss
from models.encoder import Encoder
from models.predictor2 import ActionEncoder, Predictor

import h5py
import numpy as np

DATASET = "so100-data/svla_so100_pickplace.h5"
BATCH = 32
EPOCHS = 100
SIGREG_W = 0.09
LOG_EVERY = 100
CKPT_EVERY_EPOCHS = 5
CKPT_PATH = "lewm_seq_trainonly.pt" # "lewm_seq.pt"

SEQ_LEN = 4
HISTORY = SEQ_LEN - 1
DIM = 192

SPLIT_SEED = 0
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.1

def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_episodes, _, _ = split_episodes()
    # dataset = SO100Sequences(DATASET, seq_len=SEQ_LEN)
    dataset = SO100Sequences(DATASET, seq_len=SEQ_LEN, episodes=train_episodes)
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    print(f"Training sequence windows: {len(dataset)}")

    encoder = Encoder(img_size=224, patch=16, in_ch=3, dim=DIM, depth=12, heads=3).to(device)
    action_encoder = ActionEncoder(action_dim=6, dim=DIM).to(device)
    predictor = Predictor(dim=DIM, num_frames=HISTORY).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(action_encoder.parameters())
        + list(predictor.parameters()),
        lr=5e-5,
        weight_decay=1e-3
    )

    step = 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        action_encoder.train()
        predictor.train()

        for frames, acitons in loader:
            frames = frames.to(device, non_blocking=True)
            acitons = acitons.to(device, non_blocking=True)

            batch_size, seq_len, channels, height, width = frames.shape

            z = encoder(frames.reshape(batch_size * seq_len, channels, height, width)).reshape(batch_size, seq_len, DIM)

            action_embedding = action_encoder(acitons)

            z_hat = predictor(z[:, :-1], action_embedding)
            pred_loss = nn.functional.mse_loss(z_hat, z[:, 1:])
            reg_loss = sigreg_loss(z)
            loss = pred_loss + (SIGREG_W * reg_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + list(action_encoder.parameters())
                + list(predictor.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            step += 1

            if step == 1 or step % LOG_EVERY == 0:
                with torch.no_grad():
                    z_std = z.flatten(0,1).std(dim=0)
                print(
                    f"[{datetime.now():%H:%M:%S}] "
                    f"epoch {epoch:3d}/{EPOCHS} step {step:6d} | "
                    f"pred {pred_loss.item():.6f} | "
                    f"sigreg {reg_loss.item():.5f} | "
                    f"std mean {z_std.mean().item():.4f} "
                    f"min {z_std.min().item():.4f}"
                )

        if epoch % CKPT_EVERY_EPOCHS == 0:
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "action_encoder": action_encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": step,
                },
                CKPT_PATH,
            )

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "action_encoder": action_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": EPOCHS,
            "step": step,
        },
        CKPT_PATH,
    )

    print(f"Training complete; final checkpoint: {CKPT_PATH}")
    print(f"Elapsed: {(time.time() - start) / 60:.1f} minutes")

def split_episodes():
    with h5py.File(DATASET, "r") as f:
        episodes = np.unique(f["episode_index"][:])
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(episodes)

    n_train = int(TRAIN_FRACTION * len(episodes))
    n_val = int(VAL_FRACTION * len(episodes))

    train_episodes = episodes[:n_train]
    val_episodes = episodes[n_train : n_train + n_val]
    test_episodes = episodes[n_train + n_val :]
    return train_episodes, val_episodes, test_episodes

if __name__ == "__main__":
    main()