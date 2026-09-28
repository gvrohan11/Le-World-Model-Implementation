import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SO100Sequences
from train3 import DATASET, SEQ_LEN, DIM, split_episodes
from models.encoder import Encoder
from models.predictor2 import ActionEncoder, Predictor, ProjectionHead


CKPT_PATH = "lewm_real_extra.pt"
CAMERA = "pixels_front"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    _, _, test_episodes = split_episodes()

    # Training-set action statistics are needed to reproduce train3's inputs.
    train_ds = SO100Sequences(
        DATASET,
        seq_len=SEQ_LEN,
        episodes=split_episodes()[0],
        camera=CAMERA,
    )

    # Only evaluate on the held-out test episodes.
    test_ds = SO100Sequences(
        DATASET,
        seq_len=SEQ_LEN,
        episodes=test_episodes,
        camera=CAMERA,
    )
    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    encoder = Encoder(
        img_size=224, patch=16, in_ch=3, dim=DIM, depth=12, heads=3
    ).to(DEVICE)
    action_encoder = ActionEncoder(action_dim=6, dim=DIM).to(DEVICE)
    predictor = Predictor(dim=DIM, num_frames=SEQ_LEN - 1).to(DEVICE)
    projector = ProjectionHead(DIM).to(DEVICE)
    pred_proj = ProjectionHead(DIM).to(DEVICE)

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    action_encoder.load_state_dict(checkpoint["action_encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    projector.load_state_dict(checkpoint["projector"])
    pred_proj.load_state_dict(checkpoint["pred_proj"])

    models = (encoder, action_encoder, predictor, projector, pred_proj)
    for model in models:
        model.eval()

    totals = {
        "model": 0.0,
        "persistence": 0.0,
        "zero_action": 0.0,
        "shuffled_action": 0.0,
    }
    n_values = 0
    n_windows = 0

    train_mean = train_ds.action_mean.to(DEVICE)
    train_std = train_ds.action_std.to(DEVICE)
    test_mean = test_ds.action_mean.to(DEVICE)
    test_std = test_ds.action_std.to(DEVICE)

    with torch.no_grad():
        for frames, actions_test_normalized in loader:
            frames = frames.to(DEVICE)
            actions_test_normalized = actions_test_normalized.to(DEVICE)

            # Undo test-set normalization, then apply train3's training-set
            # action normalization. This matches the checkpoint's inputs.
            raw_actions = actions_test_normalized * test_std + test_mean
            actions = (raw_actions - train_mean) / (train_std + 1e-6)

            batch, seq, channels, height, width = frames.shape
            z = encoder(
                frames.reshape(batch * seq, channels, height, width)
            )
            z = projector(z).reshape(batch, seq, DIM)

            action_embeddings = action_encoder(actions)
            predicted = pred_proj(
                predictor(z[:, :-1], action_embeddings)
            )

            targets = z[:, 1:]
            persistence = z[:, :-1]

            batch_errors = {
                "model": (predicted - targets).square(),
                "persistence": (persistence - targets).square(),
            }

            zero_action_embeddings = action_encoder(torch.zeros_like(actions))
            zero_action_prediction = pred_proj(
                predictor(z[:, :-1], zero_action_embeddings)
            )
            batch_errors["zero_action"] = (
                zero_action_prediction - targets
            ).square()

            if batch > 1:
                permutation = torch.randperm(batch, device=DEVICE)
                shuffled_actions = actions[permutation]
                shuffled_embeddings = action_encoder(shuffled_actions)
                shuffled_prediction = pred_proj(
                    predictor(z[:, :-1], shuffled_embeddings)
                )
                batch_errors["shuffled_action"] = (
                    shuffled_prediction - targets
                ).square()

            for name, errors in batch_errors.items():
                totals[name] += errors.sum().item()

            n_values += targets.numel()
            n_windows += batch

    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Test episodes: {len(test_episodes)}")
    print(f"Test sequence windows: {n_windows}")
    print("Held-out latent prediction MSE (lower is better):")
    for name, total in totals.items():
        print(f"{name:16s}: {total / n_values:.6f}")

    model_mse = totals["model"] / n_values
    persistence_mse = totals["persistence"] / n_values
    print(f"Model / persistence MSE: {model_mse / persistence_mse:.3f}")


if __name__ == "__main__":
    main()