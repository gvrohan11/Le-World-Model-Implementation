import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SO100Sequences
from losses.sigreg import sigreg_loss
from models.encoder import Encoder
from models.predictor2 import ActionEncoder, Predictor

DATASET = "so100-data/svla_so100_pickplace.h5"
BATCH = 32
EPOCHS = 100
SIGREG_W = 0.09
LOG_EVERY = 100
CKPT_EVERY_EPOCHS = 5
CKPT_PATH = "lewm_seq.pt"

SEQ_LEN = 4
HISTORY = SEQ_LEN - 1
DIM = 192

def main():
    torch.manual_seed(0)
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dataset = SO100Sequences(DATASET, seq_len=SEQ_LEN)
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
    predictor = Predictor(dim=DIM, num_frames=HIS)