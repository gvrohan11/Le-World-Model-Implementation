"""Repeated episode-held-out probes for LeWM, random ViT, and ImageNet ResNet."""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision

from models.encoder import Encoder


DATASET = "so100-data/svla_so100_pickplace.h5"
CKPT = "lewm_seq.pt" # "lewm.pt"
CAMERA = "pixels_top"
SPLIT_SEEDS = (0, 1, 2, 3, 4)
PROBE_SEEDS = (0, 1, 2)
BATCH_SIZE = 64
LINEAR_STEPS = 500
MLP_STEPS = 800
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with h5py.File(DATASET, "r") as f:
    images_np = f[f"observation/{CAMERA}"][:]
    targets_np = f["joint_pos"][:]
    episode_ids = f["episode_index"][:]

images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float().div_(255.0)
targets = torch.from_numpy(targets_np).float()
print(f"Probing on all {len(images)} frames from {len(np.unique(episode_ids))} episodes")
print(f"Device: {DEVICE}")


def make_encoder():
    return Encoder(
        img_size=224,
        patch=16,
        in_ch=3,
        dim=192,
        depth=12,
        heads=3,
    ).to(DEVICE)


lewm = make_encoder()
checkpoint = torch.load(CKPT, map_location=DEVICE, weights_only=False)
lewm.load_state_dict(checkpoint["encoder"])
random_encoder = make_encoder()

resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)
imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def recalibrate_bn(encoder, train_images):
    encoder.train()
    bn = encoder.proj_bn
    old_momentum = bn.momentum
    bn.reset_running_stats()
    bn.momentum = None

    for start in range(0, len(train_images), BATCH_SIZE):
        batch = train_images[start:start + BATCH_SIZE].to(DEVICE)
        encoder(batch)

    bn.momentum = old_momentum
    encoder.eval()


@torch.no_grad()
def embed(encoder, input_images, resnet_input=False):
    encoder.eval()
    outputs = []
    for start in range(0, len(input_images), BATCH_SIZE):
        batch = input_images[start:start + BATCH_SIZE].to(DEVICE)
        if resnet_input:
            batch = (batch - imagenet_mean) / imagenet_std
        outputs.append(encoder(batch).cpu())
    return torch.cat(outputs)


def make_episode_split(seed):
    rng = np.random.default_rng(seed)
    episodes = np.unique(episode_ids).copy()
    rng.shuffle(episodes)
    cut = int(0.8 * len(episodes))
    train_episodes, test_episodes = episodes[:cut], episodes[cut:]
    train_idx = np.flatnonzero(np.isin(episode_ids, train_episodes))
    test_idx = np.flatnonzero(np.isin(episode_ids, test_episodes))
    return torch.from_numpy(train_idx).long(), torch.from_numpy(test_idx).long()


def fit_probe(features, train_idx, test_idx, train_targets, seed, kind):
    torch.manual_seed(seed)
    x_train = features[train_idx].to(DEVICE)
    x_test = features[test_idx].to(DEVICE)
    y_train = train_targets[train_idx].to(DEVICE)
    y_test = train_targets[test_idx].to(DEVICE)

    if kind == "linear":
        model = nn.Linear(features.shape[1], train_targets.shape[1]).to(DEVICE)
        steps = LINEAR_STEPS
        lr = 1e-2
    else:
        model = nn.Sequential(
            nn.Linear(features.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, train_targets.shape[1]),
        ).to(DEVICE)
        steps = MLP_STEPS
        lr = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        residual = nn.functional.mse_loss(model(x_test), y_test)
        baseline = nn.functional.mse_loss(
            y_test.mean(0, keepdim=True).expand_as(y_test), y_test
        )
        score = 1.0 - residual / baseline.clamp_min(1e-12)
    return score.item()


def effective_rank(features):
    centered = features - features.mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return torch.exp(entropy).item()


scores = {
    name: {kind: [] for kind in ("linear", "mlp")}
    for name in ("LeWM", "Random", "ResNet18")
}
rank_scores = {"LeWM": [], "Random": [], "ResNet18": []}

for split_seed in SPLIT_SEEDS:
    train_idx, test_idx = make_episode_split(split_seed)

    train_mean = targets[train_idx].mean(0, keepdim=True)
    train_std = targets[train_idx].std(0, keepdim=True).clamp_min(1e-6)
    normalized_targets = (targets - train_mean) / train_std

    train_images = images[train_idx]
    recalibrate_bn(lewm, train_images)
    recalibrate_bn(random_encoder, train_images)

    features_by_model = {
        "LeWM": embed(lewm, images),
        "Random": embed(random_encoder, images),
        "ResNet18": embed(resnet, images, resnet_input=True),
    }

    print(
        f"split {split_seed}: train episodes={len(torch.unique(torch.from_numpy(episode_ids[train_idx.numpy()])))}, "
        f"test episodes={len(torch.unique(torch.from_numpy(episode_ids[test_idx.numpy()])))}, "
        f"frames={len(train_idx)}/{len(test_idx)}"
    )

    for name, features in features_by_model.items():
        rank_scores[name].append(effective_rank(features))
        for probe_seed in PROBE_SEEDS:
            for kind in ("linear", "mlp"):
                scores[name][kind].append(
                    fit_probe(
                        features,
                        train_idx,
                        test_idx,
                        normalized_targets,
                        seed=probe_seed,
                        kind=kind,
                    )
                )


print("\nHeld-out episode probe scores (mean ± std across splits and probe seeds):")
for name in ("LeWM", "Random", "ResNet18"):
    linear = np.asarray(scores[name]["linear"])
    mlp = np.asarray(scores[name]["mlp"])
    rank = np.asarray(rank_scores[name])
    print(
        f"{name:9s}: linear R² {linear.mean():.3f} ± {linear.std(ddof=1):.3f} | "
        f"MLP R² {mlp.mean():.3f} ± {mlp.std(ddof=1):.3f} | "
        f"effective rank {rank.mean():.1f} ± {rank.std(ddof=1):.1f}"
    )
