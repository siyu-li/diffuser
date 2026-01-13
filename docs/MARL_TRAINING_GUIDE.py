"""
DIFFUSION TRAINING WITH TRAJECTORY DATA - EXPLANATION
======================================================

This document explains how the diffusion model is trained on trajectory data
and provides recommendations for your MARL dataset.

1. HOW MAZE2D DATASET IS USED FOR DIFFUSION TRAINING
=====================================================

The training process follows these steps:

A. DATA STRUCTURE:
------------------
From your inspection output, maze2d has:
- 1566 episodes with varying lengths (95 to 3992 timesteps)
- Episode lengths: min=95, max=3993, mean=637.3
- Observation dim: 4 (x, y, vx, vy)
- Action dim: 2

B. HORIZON CONCEPT:
-------------------
The "horizon" is the LENGTH of trajectory sequences that the diffusion model learns.

Think of it as a "sliding window" over episodes:
- If horizon=128, the model learns to predict 128-timestep trajectory segments
- From a 637-step episode, you can extract many 128-step windows
  * Starting at t=0: steps [0, 128)
  * Starting at t=1: steps [1, 129)
  * Starting at t=2: steps [2, 130)
  * ... and so on

C. HOW INDICES ARE CREATED (from sequence.py):
-----------------------------------------------
```python
def make_indices(self, path_lengths, horizon):
    indices = []
    for i, path_length in enumerate(path_lengths):
        max_start = min(path_length - 1, self.max_path_length - horizon)
        if not self.use_padding:
            max_start = min(max_start, path_length - horizon)
        for start in range(max_start):
            end = start + horizon
            indices.append((i, start, end))
    return np.array(indices)
```

For an episode of length 637 with horizon=128:
- max_start = 637 - 128 = 509
- Creates 509 samples: (episode_0, 0, 128), (episode_0, 1, 129), ..., (episode_0, 509, 637)

D. TRAINING BATCH:
------------------
Each training sample is a Batch(trajectories, conditions):

1. trajectories: (horizon, transition_dim) array
   - transition_dim = action_dim + observation_dim
   - For maze2d: (128, 6) where 6 = 2 (actions) + 4 (observations)
   - For MARL: (horizon, 5) where 5 = 2 (actions) + 3 (observations: x,y,θ)
   
2. conditions: dict with timestep keys
   - GoalDataset conditions on START and END states:
     {0: start_observation, horizon-1: end_observation}
   - This tells the model: "generate a trajectory from state A to state B"

E. MAZE2D HORIZON CHOICES:
---------------------------
From config/maze2d.py:

maze2d-umaze-v1:
  - Average episode: ~150 steps
  - Horizon: 128 (85% of episode length)
  
maze2d-medium-v1:
  - Average episode: ~250 steps
  - Horizon: 256 (slightly longer than average)
  
maze2d-large-v1:
  - Average episode: ~600 steps
  - Horizon: 384 (64% of episode length)

PATTERN: Horizon is typically 50-100% of the average episode length


2. YOUR MARL DATASET ANALYSIS
==============================

Your episode length distribution:
     0 < length <=   10:   35 episodes  (3.6%)
    10 < length <=   20:  154 episodes  (15.8%)
    20 < length <=   50:  542 episodes  (55.5%) ← MAJORITY HERE
    50 < length <=  100:  249 episodes  (25.5%)
   100 < length <=  200:    2 episodes  (0.2%)

Statistics:
- Total episodes: 982
- Median length: likely around 30-40 steps
- Most episodes (55.5%) are between 20-50 steps
- 96.8% of episodes are <= 100 steps


3. HORIZON RECOMMENDATIONS FOR YOUR DATASET
============================================

OPTION 1: CONSERVATIVE - Horizon = 32
--------------------------------------
✓ Fits in 96.4% of episodes (all except 0-10 range)
✓ Allows multiple samples per episode (episodes of length 50 give ~18 samples)
✓ Safe choice, proven to work
✓ Similar ratio to maze2d-umaze (128/150 ≈ 0.85, 32/40 ≈ 0.80)

Training samples from this:
- Episodes 20-50: ~10-18 samples each × 542 episodes = ~7,000 samples
- Episodes 50-100: ~18-68 samples each × 249 episodes = ~8,500 samples
- Total: ~15,500 training samples

OPTION 2: MODERATE - Horizon = 48
----------------------------------
✓ Matches longer episodes in your dataset
✓ Still fits in most episodes (542+249 = 791 episodes, 80.5%)
✓ Allows learning longer-term behaviors
~ Episodes 20-50: fewer samples, some too short

Training samples:
- Episodes 50-100: ~2-52 samples each × 249 episodes = ~6,500 samples
- Partial coverage of 20-50 range
- Total: ~7,000-8,000 samples

OPTION 3: AGGRESSIVE - Horizon = 64
------------------------------------
✓ Learns even longer trajectories
~ Only fits well in 249 episodes (25.5%)
~ Risk: limited training data
~ Most episodes too short

Training samples:
- Episodes 50-100: ~1-36 samples each × 249 episodes = ~4,500 samples
- Total: ~4,500 samples (might be too few)


4. RECOMMENDATION: Start with Horizon = 32
===========================================

WHY:
1. Data coverage: Works with 80%+ of your episodes
2. Sample efficiency: Generates ~15,000 training samples
3. Robot planning: 32 timesteps is reasonable for robot navigation
   - If your control frequency is 10 Hz, this is 3.2 seconds of planning
   - If your control frequency is 5 Hz, this is 6.4 seconds of planning
4. Proven ratio: Similar to maze2d-umaze success
5. You can always increase later if needed

If horizon=32 trains well, you can experiment with:
- horizon=48 for longer-term planning
- horizon=24 if you need faster inference
- horizon=64 if you add more data


5. HOW TO USE YOUR MARL DATASET
================================

A. Update your config (create config/marl.py):
-----------------------------------------------
```python
import socket
from diffuser.utils import watch

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,  # ← Your choice
        'n_diffusion_steps': 100,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4),  # Smaller model for smaller dataset
        
        ## dataset
        'loader': 'datasets.MARLSequenceDataset',
        'buffer_path': 'diffuser/datasets/assets/marl_buffer.pkl',
        'normalizer': 'LimitsNormalizer',
        'use_padding': False,
        'max_path_length': 1000,
        'condition_on_goal': True,  # Condition on start and end states
        
        ## training
        'n_train_steps': 200000,  # Fewer steps for smaller dataset
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'device': 'cuda',
    },
}
```

B. Modify train.py to use MARL dataset:
----------------------------------------
```python
dataset_config = utils.Config(
    'datasets.MARLSequenceDataset',
    savepath=(args.savepath, 'dataset_config.pkl'),
    buffer_path='diffuser/datasets/assets/marl_buffer.pkl',
    horizon=args.horizon,
    normalizer=args.normalizer,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    condition_on_goal=True,
)
```


6. TRANSITION_DIM EXPLANATION
==============================

transition_dim = action_dim + observation_dim

For your MARL data:
- action_dim = 2 (linear_vel, angular_vel)
- observation_dim = 3 (x, y, theta)
- transition_dim = 5

The model concatenates actions and observations:
[action_0, action_1, obs_0, obs_1, obs_2]
[v_linear, v_angular, x, y, theta]

This is the format the diffusion model learns to denoise.


7. KEY PARAMETERS SUMMARY
==========================

Parameter              | maze2d-umaze | Your MARL Dataset
-----------------------|--------------|-------------------
Average episode length | ~150         | ~35-40
Horizon               | 128          | 32 (recommended)
Horizon/Episode ratio | 0.85         | 0.80-0.91
Observation dim       | 4            | 3
Action dim            | 2            | 2
Transition dim        | 6            | 5
n_diffusion_steps     | 64           | 100 (recommended)
Batch size            | 32           | 32
Training samples      | ~900K        | ~15K
Conditions            | {0, 127}     | {0, 31} (start/end)


8. NEXT STEPS
=============

1. Create config/marl.py with horizon=32
2. Test the MARLSequenceDataset loads correctly
3. Start training with small n_train_steps (e.g., 10000) to verify
4. Monitor training loss and samples
5. Adjust horizon if needed based on results

"""


if __name__ == "__main__":
    print(__doc__)
