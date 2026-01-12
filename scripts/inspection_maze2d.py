"""
Inspect maze2d dataset structure to understand trajectories and conditions
Usage: python scripts/inspect_dataset.py
"""

import numpy as np
import torch
import diffuser.utils as utils
from diffuser.datasets.sequence import SequenceDataset
import pdb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'  # Small maze for quick inspection
    config: str = 'config.maze2d'

args = Parser().parse_args('diffusion')

#-----------------------------------------------------------------------------#
#------------------------------- load dataset --------------------------------#
#-----------------------------------------------------------------------------#

print("="*80)
print("LOADING DATASET")
print("="*80)

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

dataset = dataset_config()

print(f"\nDataset: {args.dataset}")
print(f"Dataset class: {dataset.__class__.__name__}")
print(f"Number of episodes: {dataset.n_episodes}")
print(f"Total timesteps: {dataset.n_steps}")
print(f"Observation dim: {dataset.observation_dim}")
print(f"Action dim: {dataset.action_dim}")
print(f"Transition dim: {dataset.transition_dim}")
print(f"Horizon: {dataset.horizon}")

#-----------------------------------------------------------------------------#
#---------------------------- inspect raw data -------------------------------#
#-----------------------------------------------------------------------------#

print("\n" + "="*80)
print("RAW DATA STRUCTURE (Before batching)")
print("="*80)

# Access the underlying fields
print("\n1. Dataset.fields structure:")
print(f"   Type: {type(dataset.fields)}")
print(f"   Keys: {dataset.fields._fields}")

print("\n2. Raw observations:")
print(f"   Shape: {dataset.fields.observations.shape}")
print(f"   Type: {dataset.fields.observations.dtype}")
print(f"   Min: {dataset.fields.observations.min():.3f}")
print(f"   Max: {dataset.fields.observations.max():.3f}")
print(f"   First 5 observations:")
print(f"   {dataset.fields.observations[:5]}")

print("\n3. Raw actions:")
print(f"   Shape: {dataset.fields.actions.shape}")
print(f"   Type: {dataset.fields.actions.dtype}")
print(f"   Min: {dataset.fields.actions.min():.3f}")
print(f"   Max: {dataset.fields.actions.max():.3f}")
print(f"   First 5 actions:")
print(f"   {dataset.fields.actions[:5]}")

print("\n4. Raw rewards:")
print(f"   Shape: {dataset.fields.rewards.shape}")
print(f"   Type: {dataset.fields.rewards.dtype}")
print(f"   Min: {dataset.fields.rewards.min():.3f}")
print(f"   Max: {dataset.fields.rewards.max():.3f}")
print(f"   Sum: {dataset.fields.rewards.sum():.3f}")

print("\n5. Episode boundaries (path_lengths):")
print(f"   Number of episodes: {len(dataset.path_lengths)}")
print(f"   First 10 episode lengths: {dataset.path_lengths[:10]}")
print(f"   Min episode length: {min(dataset.path_lengths)}")
print(f"   Max episode length: {max(dataset.path_lengths)}")
print(f"   Mean episode length: {np.mean(dataset.path_lengths):.1f}")

#-----------------------------------------------------------------------------#
#------------------------ inspect single batch item --------------------------#
#-----------------------------------------------------------------------------#

print("\n" + "="*80)
print("SINGLE BATCH ITEM (What model sees during training)")
print("="*80)

# Get one item from dataset
item_idx = 0
batch_item = dataset[item_idx]

print(f"\n1. Batch item keys:")
print(f"   {batch_item.keys()}")

print(f"\n2. Trajectories:")
trajectories = batch_item['trajectories']
print(f"   Shape: {trajectories.shape}")
print(f"   Type: {type(trajectories)}")
print(f"   Dtype: {trajectories.dtype}")
print(f"   Interpretation: (horizon={trajectories.shape[0]}, transition_dim={trajectories.shape[1]})")
print(f"   transition_dim = action_dim({dataset.action_dim}) + observation_dim({dataset.observation_dim})")

print(f"\n   First 3 timesteps of trajectory:")
for t in range(min(3, trajectories.shape[0])):
    action_part = trajectories[t, :dataset.action_dim]
    obs_part = trajectories[t, dataset.action_dim:]
    print(f"   t={t}: action={action_part.numpy()}, observation={obs_part.numpy()}")

print(f"\n3. Conditions:")
conditions = batch_item['conditions']
print(f"   Type: {type(conditions)}")
print(f"   Keys (timesteps): {list(conditions.keys())}")

for timestep, condition_value in conditions.items():
    print(f"   Condition at t={timestep}:")
    print(f"      Shape: {condition_value.shape}")
    print(f"      Value: {condition_value.numpy()}")
    print(f"      Interpretation: Initial state (x, y, ?, ?) in maze")

#-----------------------------------------------------------------------------#
#---------------------- inspect multiple batch items -------------------------#
#-----------------------------------------------------------------------------#

print("\n" + "="*80)
print("MULTIPLE BATCH ITEMS (Understanding sliding window)")
print("="*80)

print(f"\nDataset creates sliding windows of horizon={dataset.horizon}")
print(f"Total indices available: {len(dataset.indices)}")
print(f"First 10 indices: {dataset.indices[:10]}")

# Show several consecutive items
print(f"\nShowing 3 consecutive batch items:")
for i in range(3):
    item = dataset[i]
    traj = item['trajectories']
    cond = item['conditions']
    
    print(f"\nBatch item {i}:")
    print(f"  Trajectory shape: {traj.shape}")
    print(f"  First observation (t=0): {traj[0, dataset.action_dim:].numpy()}")
    print(f"  Last observation (t={dataset.horizon-1}): {traj[-1, dataset.action_dim:].numpy()}")
    print(f"  Condition (start state): {cond[0].numpy()}")

#-----------------------------------------------------------------------------#
#------------------------- inspect batched data ------------------------------#
#-----------------------------------------------------------------------------#

print("\n" + "="*80)
print("BATCHED DATA (What goes into model during training)")
print("="*80)

# Simulate what happens in training
batch_size = 4
batch_items = [dataset[i] for i in range(batch_size)]

# Stack trajectories
trajectories_batch = torch.stack([item['trajectories'] for item in batch_items], dim=0)
print(f"\n1. Batched trajectories:")
print(f"   Shape: {trajectories_batch.shape}")
print(f"   Interpretation: (batch_size={batch_size}, horizon={dataset.horizon}, transition_dim={dataset.transition_dim})")

# Collect conditions
conditions_batch = {}
for item in batch_items:
    for t, cond in item['conditions'].items():
        if t not in conditions_batch:
            conditions_batch[t] = []
        conditions_batch[t].append(cond)

for t in conditions_batch:
    conditions_batch[t] = torch.stack(conditions_batch[t], dim=0)

print(f"\n2. Batched conditions:")
print(f"   Keys (timesteps): {list(conditions_batch.keys())}")
for t, cond in conditions_batch.items():
    print(f"   Condition at t={t}:")
    print(f"      Shape: {cond.shape}")
    print(f"      Interpretation: (batch_size={batch_size}, observation_dim={dataset.observation_dim})")

#-----------------------------------------------------------------------------#
#-------------------- show what goes into diffusion --------------------------#
#-----------------------------------------------------------------------------#

print("\n" + "="*80)
print("DIFFUSION MODEL INPUT/OUTPUT")
print("="*80)

# Simulate one training step
print("\nSimulating one training step...")

# Get batch
x = trajectories_batch  # (B, H, transition_dim)
cond = conditions_batch  # {0: (B, obs_dim)}

print(f"\n1. Input to diffusion.loss():")
print(f"   x shape: {x.shape}")
print(f"   cond keys: {list(cond.keys())}")
print(f"   cond[0] shape: {cond[0].shape}")

# Simulate adding noise (forward diffusion)
t = torch.randint(0, 100, (batch_size,)).long()
print(f"\n2. Random timesteps sampled:")
print(f"   t: {t}")

noise = torch.randn_like(x)
print(f"\n3. Random noise:")
print(f"   noise shape: {noise.shape}")

# This is what happens in q_sample
sqrt_alpha = torch.rand(batch_size, 1, 1)  # Simplified
x_noisy = sqrt_alpha * x + (1 - sqrt_alpha) * noise
print(f"\n4. Noisy trajectory (q_sample result):")
print(f"   x_noisy shape: {x_noisy.shape}")
print(f"   Interpretation: Original trajectory + noise at timestep t")

#-----------------------------------------------------------------------------#
#----------------------- maze2d specific visualization -----------------------#
#-----------------------------------------------------------------------------#

print("\n" + "="*80)
print("MAZE2D SPECIFIC INTERPRETATION")
print("="*80)

print("\nFor maze2d navigation:")
print("  - Observation: [x, y, goal_x, goal_y] - 4D position in maze")
print("  - Action: [dx, dy] - 2D velocity command")
print("  - Trajectory: sequence of (action, observation) pairs over horizon")
print("  - Condition: starting position [x, y, goal_x, goal_y]")

print("\nExample trajectory interpretation:")
traj = trajectories_batch[0]  # First item in batch
print(f"  Start state: {traj[0, dataset.action_dim:].numpy()}")
print(f"  Start action: {traj[0, :dataset.action_dim].numpy()}")
print(f"  End state: {traj[-1, dataset.action_dim:].numpy()}")
print(f"  Path length: {dataset.horizon} steps")

#-----------------------------------------------------------------------------#
#---------------------------- summary diagram --------------------------------#
#-----------------------------------------------------------------------------#

print("\n" + "="*80)
print("SUMMARY DIAGRAM")
print("="*80)

print("""
DATA FLOW:

1. Raw Dataset (from D4RL):
   observations: (N, 4)  [all timesteps concatenated]
   actions: (N, 2)       [all timesteps concatenated]
   rewards: (N,)
   
2. Episodic Segmentation:
   Episode 1: obs[0:150], actions[0:150], ...
   Episode 2: obs[150:300], actions[150:300], ...
   ...
   
3. Sliding Window (SequenceDataset):
   Window 0: [obs[0:32], actions[0:32]]
   Window 1: [obs[1:33], actions[1:33]]
   Window 2: [obs[2:34], actions[2:34]]
   ...
   
4. Single Batch Item (dataset[i]):
   trajectories: (horizon=32, transition_dim=6)
      - Contains: [action_0, obs_0, action_1, obs_1, ..., action_31, obs_31]
      - Actually shaped as: [[a0, o0], [a1, o1], ..., [a31, o31]]
   conditions: {0: (4,)}
      - Initial state: obs_0
      
5. Batched for Training:
   trajectories: (batch=32, horizon=32, transition_dim=6)
   conditions: {0: (batch=32, obs_dim=4)}
   
6. During Training (inside diffusion.loss()):
   - Sample random noise: ε ~ N(0, I)
   - Sample random timestep: t ~ Uniform(0, T-1)
   - Add noise to trajectories: x_noisy = sqrt(α_t)*x + sqrt(1-α_t)*ε
   - Model predicts: ε_pred = TemporalUnet(x_noisy, cond, t)
   - Compute loss: ||ε_pred - ε||²
   
7. During Sampling (diffusion.conditional_sample()):
   - Start with pure noise: x_T ~ N(0, I)
   - For t = T-1 down to 0:
       - Predict noise: ε_pred = TemporalUnet(x_t, cond, t)
       - Remove noise: x_{t-1} = denoise(x_t, ε_pred)
   - Final x_0 is the generated trajectory
""")

print("\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)
print("\nTo run this script:")
print("  python scripts/inspect_dataset.py")
print("\nYou can also enter interactive mode by adding 'pdb.set_trace()' anywhere above.")