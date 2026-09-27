from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_ID = "yangxinye/real_so101_record_v1"
ROOT = "so101-real-extra"
OUTPUT = "so100-data/real_so101_extra.h5"
CAMERA_KEY = "observation.images.cam_front"
IMAGE_KEY = "pixels_front"
IMAGE_SIZE = 224


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


ds = LeRobotDataset(REPO_ID, root=ROOT, return_uint8=True)
num_frames = len(ds)

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)

with h5py.File(OUTPUT, "w") as f:
    observation = f.create_group("observation")
    observation.create_dataset(
        IMAGE_KEY,
        shape=(num_frames, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
        chunks=(1, IMAGE_SIZE, IMAGE_SIZE, 3),
        compression="lzf",
    )
    f.create_dataset("action", shape=(num_frames, 6), dtype=np.float32)
    f.create_dataset("joint_pos", shape=(num_frames, 6), dtype=np.float32)
    f.create_dataset("episode_index", shape=(num_frames,), dtype=np.int64)
    f.create_dataset("timestep", shape=(num_frames,), dtype=np.int64)

    for i in range(num_frames):
        sample = ds[i]

        image = sample[CAMERA_KEY]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"Unexpected image shape: {tuple(image.shape)}")

        image = image.float()
        if image.max().item() <= 1.0:
            image = image * 255.0

        image = F.interpolate(
            image.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        image = (
            image[0]
            .clamp(0, 255)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        action = as_numpy(sample["action"]).astype(np.float32)
        state = as_numpy(sample["observation.state"]).astype(np.float32)

        if action.shape != (6,) or state.shape != (6,):
            raise ValueError(
                f"Unexpected action/state shapes: {action.shape}, {state.shape}"
            )

        observation[IMAGE_KEY][i] = image
        f["action"][i] = action
        f["joint_pos"][i] = state
        f["episode_index"][i] = int(as_numpy(sample["episode_index"]).item())
        f["timestep"][i] = int(as_numpy(sample["frame_index"]).item())

        if (i + 1) % 1000 == 0:
            print(f"Converted {i + 1}/{num_frames} frames")

print(f"Wrote {OUTPUT}: {num_frames} frames, {ds.num_episodes} episodes")