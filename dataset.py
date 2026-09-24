import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

'''
Build the data loader - this reads real (frame, next-frame, action) examples from the dataset
and hands them to the model in batches, replacing the fake torch.randn data
'''

class SO100Pairs(Dataset):
    def __init__(self, h5_path, camera="pixels_top", gap=8):
        self.h5_path = h5_path
        self.camera = camera
        self.file = None
        with h5py.File(h5_path, "r") as f:
            ep = f["episode_index"][:]
            ts = f["timestep"][:]
            all_actions = f["action"][:]
        self.action_mean = torch.from_numpy(all_actions.mean(0)).float()
        self.action_std = torch.from_numpy(all_actions.std(0)).float()
        pairs = []
        for e in np.unique(ep):
            rows = np.where(ep == e)[0]
            rows = rows[np.argsort(ts[rows])]
            # pairs.extend(zip(rows[:-1], rows[1:]))
            if len(rows) > gap:
                pairs.extend(zip(rows[:-gap], rows[gap:]))
        self.pairs = np.array(pairs)
        # self.valid = np.where(ep[:-1] == ep[1:][0])[0]

    def __len__(self):
        return len(self.pairs)

    def _f(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        return self.file

    def __getitem__(self, key):
        t, t1 = self.pairs[key]
        f = self._f()
        img_t = f["observation"][self.camera][t]
        img_t1 = f["observation"][self.camera][t1]
        action = torch.from_numpy(f["action"][t]).float()
        action = (action - self.action_mean) / (self.action_std + 1e-6)
        frame = torch.from_numpy(img_t).permute(2,0,1).float() / 255.0
        next_frame = torch.from_numpy(img_t1).permute(2,0,1).float() / 255.0
        return frame, next_frame, action

class SO100Sequences(Dataset):
    def __init__(self, h5_path, seq_len=4, camera="pixels_top"):
        self.h5_path = h5_path
        self.seq_len = seq_len
        self.camera = camera
        self.file = None

        with h5py.File(h5_path, "r") as f:
            episode_ids = f["episode_index"][:]
            timesteps = f["timestep"][:]
            all_actions = f["action"][:]

        self.action_mean = torch.from_numpy(all_actions.mean(axis=0)).float()
        self.action_std = torch.from_numpy(all_actions.std(axis=0)).float()

        windows = []
        for episode in np.unique(episode_ids):
            rows = np.flatnonzero(episode_ids == episode)
            rows = rows[np.argsort(timesteps[rows])]

            for start in range(len(rows) - seq_len + 1):
                window = rows[start : start + seq_len]
                if np.all(np.diff(timesteps[window]) == 1):
                    windows.append(window)

        self.windows = np.asarray(windows, dtype=np.int64)

    def __len__(self):
        return len(self.windows)

    def _file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
        return self.file

    def __getitem__(self, index):
        rows = self.windows[index]
        f = self._file()

        frame_dataset = f["observation"][self.camera]
        frames_np = np.stack([frame_dataset[row] for row in rows])

        action_dataset = f["action"]
        actions_np = np.stack([action_dataset[row] for row in rows[:-1]])

        frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0 # Why go in 0,3,1,2 order?
        actions = torch.from_numpy(actions_np).float()
        actions = (actions - self.action_mean) / (self.action_std + 1e-6)

        return frames, actions

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = SO100Pairs("so100-data/svla_so100_pickplace.h5")
    print(f"Training Pairs: {len(ds)}")
    frame, next_frame, action = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    print(f"Frames: {tuple(frame.shape)}")
    print(f"Actions: {tuple(action.shape)}")
    print(f"Pixels: {round(frame.min().item(), 2)} to {round(frame.max().item(), 2)}")