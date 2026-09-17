## Training on macbook

I tried to run the training loop (train2.py) for 8 hours on my macbook, here's what I noticed:

Macbook Speed:
- 10 seconds per training step (one batch of 62 pairs)
- 6 steps per minute, 355 steps per hour
- 12 images / second on the VIT - each step encodes 124 images (62 pairs * 2 frames: current & next)

The model it's pushing through:
- 6M parameters (encoder 5.5M + predictor 0.46M) fully updated once per step
- Roughly .5 TFLOP of compute per step (forward + backward)

This means: 50k steps / 355 steps per hr = 140 hours, or 6 days