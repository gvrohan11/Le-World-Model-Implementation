import torch
import torch.nn as nn
from models.encoder import Encoder
from models.predictor import Predictor
from losses.sigreg import sigreg_loss

'''
Combining sigreg, encoder, and predictor into 1
Wire everything into a single training step: take a real frame, the next frame, encode both, have the predictor guess the next embedding
from the current one plus the action, and combine 2 losses: how wrong the prediction was, plus SIGReg keeping the embeddings healthy
'''

def training_step(encoder, predictor, frame, next_frame, action, sigreg_weight=1.0):
    z = encoder(frame)
    z_next = encoder(next_frame)
    z_next_hat = predictor(z, action)
    pred_loss = nn.functional.mse_loss(z_next_hat, z_next)
    reg_loss = sigreg_loss(z) # regularized loss
    total = pred_loss + (sigreg_weight * reg_loss)
    return total, pred_loss, reg_loss

if __name__ == "__main__":
    enc = Encoder()
    pred = Predictor()
    frame = torch.randn(16, 3, 64, 64)
    next_frame = torch.randn(16, 3, 64, 64)
    action = torch.randn(16, 7)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)
    for i in range(4):
        total, pl, rl = training_step(enc, pred, frame, next_frame, action)
        opt.zero_grad()
        total.backward()
        opt.step()
        print(f"Step {i}: total={total.item():.4f} pred={pl.item():.4f} sigreg={rl.item():.4f}")
