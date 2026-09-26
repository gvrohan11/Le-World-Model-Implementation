---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- robotics
- video
- time-series
- tabular
pretty_name: real_so101_record_v1
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
---

# real_so101_record_v1

A real-world LeRobot-format dataset collected on a SO101/SO100-style follower arm with three camera views.

## Summary

This dataset contains robot demonstrations recorded with `lerobot-record` for tabletop manipulation tasks. It includes robot states, actions, multi-view images, metadata, parquet trajectories, and video files.

## Task

Example task:
- Pick up any purple cube from the white area and place it in cell 8

## Setup

- Robot: SO101 / SO100-style follower arm
- Cameras:
  - cam_hand
  - cam_front
  - cam_side

## Structure

```text
data/
meta/
videos/
README.md
```

## Notes

This dataset follows the standard LeRobot dataset format and is intended for imitation learning, policy evaluation, and robotics research.

## License

Apache-2.0