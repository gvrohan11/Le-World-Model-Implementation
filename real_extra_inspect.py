from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "yangxinye/real_so101_record_v1",
    root="so101-real-extra",
)

print("episodes:", ds.num_episodes)
print("frames:", ds.num_frames)

sample = ds[0]
for key, value in sample.items():
    shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
    print(key, shape)