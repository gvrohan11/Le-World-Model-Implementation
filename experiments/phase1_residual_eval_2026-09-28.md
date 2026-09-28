# Phase 1 - Residual LeWM transition evaluation

## Model and dada

- Checkpoitn: 'lewm_real_extra_residual.pt'
- Training: 100 epochs; 128.8 mins
- Dataset: 'so100-data/real_so101_extra.h5' - 41717 frames from 90 episodes
- Input: 'pixels_front', sequence length 4
- Episode splt: 62 train / 9 validation / 19 test
- Predictor uses residual form: current latent + preducted latent ciirrection 
- Evaluation compares the learned prediction with persistence (predicting no latent change)

## Results

Validation episodes ("val"):
Overall MSE: LeWM: 0.003003, persistence: 0.002734, LeWM/persistence = 1.098
Largest-change quarter: LeWM 0.009301; persistence 0.009383

Held-out test episodes ("test"):
Overall MSE: LeWM 0.002730; persistence 0.002675; LeWM/persistence = 1.021
Largest-change quarter: LeWM 0.008709; persistence 0.009144
Middle-transition MSE by action offset: previous 0.002724; current 0.002729; next 0.002772; persistence 0.002677
Action ablation MSE: correct actions 0.002730; shuffled actions 0.002964; zero actions 0.003600

## Interpretation and decision

Predictor is worse than persistence for held-out episodes (~2%). It's better on the largest-change quarter, and correct actions are better than no actions. Therefore, the model uses action info, but we don't know if it predicts better than persistence or controls the bot.

The results are a model diagnostic, not a ball-in-cup task evaluation.

The predictor is slightly worse than persistence overall on held-out episodes (about 2%). It is slightly better on the largest-change quarter, and correct actions outperform shuffled or zero actions. This is evidence that the model uses action information, but not evidence that it consistently predicts better than persistence or controls the robot.

The “largest-change” subset is selected by latent change, not verified physical object movement. These results are a model diagnostic, not a ball-in-cup task evaluation. Keep this checkpoint as a candidate baseline; do not claim the overall persistence baseline was beaten.

- Task content: pick orange/purple cubes from a white area and place them in numbered grid cells.
- These episodes are generic manipulation pretraining data, not ball-in-cup policy demonstrations.