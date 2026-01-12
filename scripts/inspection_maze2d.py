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
print(f"Total timesteps: {dataset.fields.n_steps}")
print(f"Observation dim: {dataset.observation_dim}")
print(f"Action dim: {dataset.action_dim}")
print(f"Transition dim: {dataset.action_dim + dataset.observation_dim}")
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
print(f"   Keys: {dataset.fields.keys}")

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

print(f"\n1. Batch item structure:")
print(f"   Type: {type(batch_item)}")
print(f"   Fields: {batch_item._fields}")

print(f"\n2. Trajectories:")
trajectories = batch_item.trajectories  # Access namedtuple field directly
transition_dim = dataset.action_dim + dataset.observation_dim
print(f"   Shape: {trajectories.shape}")
print(f"   Type: {type(trajectories)}")
print(f"   Dtype: {trajectories.dtype}")
print(f"   Interpretation: (horizon={trajectories.shape[0]}, transition_dim={trajectories.shape[1]})")
print(f"   transition_dim = action_dim({dataset.action_dim}) + observation_dim({dataset.observation_dim})")

print(f"\n   First 3 timesteps of trajectory:")
for t in range(min(3, trajectories.shape[0])):
    action_part = trajectories[t, :dataset.action_dim]
    obs_part = trajectories[t, dataset.action_dim:]
    print(f"   t={t}: action={action_part}, observation={obs_part}")

print(f"\n3. Conditions:")
conditions = batch_item.conditions  # Access namedtuple field directly
print(f"   Type: {type(conditions)}")
print(f"   Keys (timesteps): {list(conditions.keys())}")

for timestep, condition_value in conditions.items():
    print(f"   Condition at t={timestep}:")
    print(f"      Shape: {condition_value.shape}")
    print(f"      Value: {condition_value}")
    print(f"      Interpretation: Initial state [x_pos, y_pos, x_vel, y_vel]")


#-----------------------------------------------------------------------------#
#------------------------- inspect batched data ------------------------------#
#-----------------------------------------------------------------------------#

print("\n" + "="*80)
print("BATCHED DATA (What goes into model during training)")
print("="*80)

# Simulate what happens in training
batch_size = 4
batch_items = [dataset[i] for i in range(batch_size)]

# Stack trajectories (convert numpy to torch tensors like DataLoader does)
trajectories_batch = torch.stack([torch.from_numpy(item.trajectories) for item in batch_items], dim=0)
transition_dim = dataset.action_dim + dataset.observation_dim
print(f"\n1. Batched trajectories:")
print(f"   Shape: {trajectories_batch.shape}")
print(f"   Interpretation: (batch_size={batch_size}, horizon={dataset.horizon}, transition_dim={transition_dim})")

# Collect conditions
conditions_batch = {}
for item in batch_items:
    for t, cond in item.conditions.items():
        if t not in conditions_batch:
            conditions_batch[t] = []
        conditions_batch[t].append(torch.from_numpy(cond))

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

