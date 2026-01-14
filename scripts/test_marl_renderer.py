"""
Test script for the MARL robot renderer.
Visualizes sample trajectories from the MARL dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from diffuser.utils.rendering import MARLRobotRenderer
from diffuser.datasets.marl import MARLSequenceDataset

# Load the dataset
print("Loading MARL dataset...")
dataset = MARLSequenceDataset(
    buffer_path='diffuser/datasets/assets/marl_buffer.pkl',
    horizon=32,
    normalizer='LimitsNormalizer',
    use_padding=False,
    condition_on_goal=True,
)

print(f"Dataset loaded with {len(dataset)} samples")
print(f"Observation dim: {dataset.observation_dim}")
print(f"Action dim: {dataset.action_dim}")

# Create renderer
renderer = MARLRobotRenderer(
    env='marl',
    workspace_bounds=(0, 12, 0, 12),  # Adjust based on your workspace
    observation_dim=dataset.observation_dim
)

# Sample a few trajectories from the dataset
n_samples = 8
sample_indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

print(f"\nRendering {len(sample_indices)} sample trajectories...")

paths = []
for idx in sample_indices:
    batch = dataset[idx]
    # batch is a namedtuple with 'trajectories' and 'conditions'
    # trajectories shape: (horizon, action_dim + observation_dim)
    trajectories = batch.trajectories
    
    # Extract observations from trajectories
    # Assuming observations are at the end of each timestep
    observations = trajectories[:, dataset.action_dim:]
    
    # Denormalize for visualization
    observations = dataset.normalizer.unnormalize(observations, 'observations')
    
    paths.append(observations)
    print(f"Sample {idx}: observation shape = {observations.shape}")

# Render all trajectories in a grid
output_path = 'logs/marl/sample_trajectories.png'
renderer.composite(output_path, paths, ncol=4)

print(f"\n✓ Visualization saved to {output_path}")

# Also create a single trajectory visualization with more details
print("\nRendering detailed single trajectory...")
sample_obs = paths[0]
single_output_path = 'logs/marl/single_trajectory.png'

img = renderer.renders(
    sample_obs,
    title='Sample Robot Trajectory',
    show_orientation=True
)

import imageio
imageio.imsave(single_output_path, img)
print(f"✓ Single trajectory saved to {single_output_path}")

# Optional: Create a video of the trajectory
print("\nCreating trajectory video...")
video_path = 'logs/marl/trajectory_video.mp4'
renderer.render_rollout(video_path, sample_obs, fps=10)
print(f"✓ Video saved to {video_path}")

print("\n✓ All visualizations complete!")
