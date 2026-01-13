"""
Script to convert multi-robot YAML data to PyTorch dataset format compatible with SequenceDataset.

The YAML data contains 5 robots with poses (x, y, theta) and actions (linear_vel, angular_vel).
This script splits the data into individual robot trajectories and segments them based on goal completion.
"""

import yaml
import numpy as np
import pickle
import os
from pathlib import Path


def load_yaml_data(yaml_path):
    """Load data from YAML file."""
    print(f"Loading data from {yaml_path}...")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    print(f"Loaded {len(data)} timesteps")
    return data


def extract_robot_trajectories(yaml_data, num_robots=5):
    """
    Extract individual robot trajectories from multi-robot data.
    
    Args:
        yaml_data: Dictionary with timestep keys containing robot data
        num_robots: Number of robots in the data
        
    Returns:
        List of trajectories, where each trajectory is a dict with:
        - observations: array of shape (T, 3) containing [x, y, theta]
        - actions: array of shape (T, 2) containing [linear_vel, angular_vel]
        - goals: array of shape (T,) containing boolean goal reached flags
    """
    # Sort timesteps to ensure chronological order
    # Handle both int and string keys
    timesteps = sorted(yaml_data.keys())
    num_timesteps = len(timesteps)
    
    # Initialize arrays for all robots
    all_robot_data = []
    for robot_idx in range(num_robots):
        robot_data = {
            'observations': [],
            'actions': [],
            'goals': []
        }
        
        for t in timesteps:
            step_data = yaml_data[t]
            
            # Extract pose (x, y, theta) for this robot
            pose = step_data['poses'][robot_idx]
            robot_data['observations'].append(pose)
            
            # Extract action (linear_vel, angular_vel) for this robot
            action = step_data['actions'][robot_idx]
            robot_data['actions'].append(action)
            
            # Extract goal reached flag
            goal = step_data['goals'][robot_idx]
            robot_data['goals'].append(goal)
        
        # Convert lists to numpy arrays
        robot_data['observations'] = np.array(robot_data['observations'], dtype=np.float32)
        robot_data['actions'] = np.array(robot_data['actions'], dtype=np.float32)
        robot_data['goals'] = np.array(robot_data['goals'], dtype=bool)
        
        all_robot_data.append(robot_data)
    
    return all_robot_data


def segment_trajectories(robot_data, min_length=3):
    """
    Segment a robot's full trajectory into episodes based on goal completion.
    
    A trajectory starts when:
    - It's the first timestep (t=0), OR
    - The previous timestep had goal=True (goal was just reached)
    
    A trajectory ends when:
    - goal=True (goal is reached), OR
    - It's the last timestep
    
    Args:
        robot_data: Dictionary with 'observations', 'actions', 'goals' arrays
        min_length: Minimum trajectory length to keep (default: 3, filters out length 1 and 2)
        
    Returns:
        List of episode dictionaries, each containing:
        - observations: (T, 3) array
        - actions: (T, 2) array
        - terminals: (T, 1) array (1.0 at goal, 0.0 otherwise)
    """
    goals = robot_data['goals']
    observations = robot_data['observations']
    actions = robot_data['actions']
    
    episodes = []
    start_idx = 0
    
    for t in range(1, len(goals)):
        # Check if previous step reached goal (end of episode)
        if goals[t - 1]:
            # Create episode from start_idx to t (exclusive)
            episode_length = t - start_idx
            if episode_length >= min_length:
                episode = {
                    'observations': observations[start_idx:t].copy(),
                    'actions': actions[start_idx:t].copy(),
                    'terminals': np.zeros((episode_length, 1), dtype=np.float32),
                }
                # Mark the last step as terminal
                episode['terminals'][-1] = 1.0
                episodes.append(episode)
            
            # Start new episode
            start_idx = t
    
    # Handle the last episode
    if start_idx < len(goals):
        episode_length = len(goals) - start_idx
        if episode_length >= min_length:
            episode = {
                'observations': observations[start_idx:].copy(),
                'actions': actions[start_idx:].copy(),
                'terminals': np.zeros((episode_length, 1), dtype=np.float32),
            }
            # Mark as terminal if goal was reached
            if goals[-1]:
                episode['terminals'][-1] = 1.0
            episodes.append(episode)
    
    return episodes


def create_replay_buffer_dict(all_episodes, max_path_length=None):
    """
    Create a dictionary in ReplayBuffer format from list of episodes.
    
    Args:
        all_episodes: List of episode dictionaries
        max_path_length: Maximum length to pad/truncate episodes to
        
    Returns:
        Dictionary with fields compatible with ReplayBuffer
    """
    n_episodes = len(all_episodes)
    
    # Determine max_path_length if not provided
    if max_path_length is None:
        max_path_length = max(len(ep['observations']) for ep in all_episodes)
    
    # Get dimensions from first episode
    observation_dim = all_episodes[0]['observations'].shape[-1]
    action_dim = all_episodes[0]['actions'].shape[-1]
    
    # Initialize arrays
    buffer_dict = {
        'observations': np.zeros((n_episodes, max_path_length, observation_dim), dtype=np.float32),
        'actions': np.zeros((n_episodes, max_path_length, action_dim), dtype=np.float32),
        'terminals': np.zeros((n_episodes, max_path_length, 1), dtype=np.float32),
        'rewards': np.zeros((n_episodes, max_path_length, 1), dtype=np.float32),
        'timeouts': np.zeros((n_episodes, max_path_length, 1), dtype=np.float32),
        'path_lengths': np.zeros(n_episodes, dtype=np.int32),
    }
    
    # Fill in episode data
    for i, episode in enumerate(all_episodes):
        path_length = len(episode['observations'])
        
        # Truncate if necessary
        if path_length > max_path_length:
            path_length = max_path_length
        
        buffer_dict['observations'][i, :path_length] = episode['observations'][:path_length]
        buffer_dict['actions'][i, :path_length] = episode['actions'][:path_length]
        buffer_dict['terminals'][i, :path_length] = episode['terminals'][:path_length]
        
        # Set reward to 1.0 when goal is reached
        buffer_dict['rewards'][i, :path_length] = episode['terminals'][:path_length]
        
        buffer_dict['path_lengths'][i] = path_length
    
    # Create next_observations (shifted observations)
    buffer_dict['next_observations'] = np.zeros_like(buffer_dict['observations'])
    for i in range(n_episodes):
        path_length = buffer_dict['path_lengths'][i]
        if path_length > 1:
            buffer_dict['next_observations'][i, :path_length-1] = buffer_dict['observations'][i, 1:path_length]
            buffer_dict['next_observations'][i, path_length-1] = buffer_dict['observations'][i, path_length-1]
    
    return buffer_dict


def save_buffer_dict(buffer_dict, output_path):
    """Save buffer dictionary to pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(buffer_dict, f)
    
    print(f"\nSaved buffer to {output_path}")
    
    # Print statistics
    n_episodes = len(buffer_dict['path_lengths'])
    total_steps = buffer_dict['path_lengths'].sum()
    print(f"\nDataset Statistics:")
    print(f"  Number of episodes: {n_episodes}")
    print(f"  Total timesteps: {total_steps}")
    print(f"  Average episode length: {total_steps / n_episodes:.1f}")
    print(f"  Min episode length: {buffer_dict['path_lengths'].min()}")
    print(f"  Max episode length: {buffer_dict['path_lengths'].max()}")
    print(f"  Observation dim: {buffer_dict['observations'].shape[-1]}")
    print(f"  Action dim: {buffer_dict['actions'].shape[-1]}")


def main():
    """Main conversion pipeline."""
    # Paths
    yaml_path = "diffuser/datasets/assets/marl_data.yml"
    output_path = "diffuser/datasets/assets/marl_buffer.pkl"
    
    # Load YAML data
    yaml_data = load_yaml_data(yaml_path)
    
    # Extract individual robot trajectories
    print("\nExtracting robot trajectories...")
    all_robot_data = extract_robot_trajectories(yaml_data, num_robots=5)
    
    # Segment into episodes for each robot
    print("\nSegmenting trajectories by goal completion...")
    print("(Filtering out episodes with length < 3)")
    all_episodes = []
    for robot_idx, robot_data in enumerate(all_robot_data):
        episodes = segment_trajectories(robot_data, min_length=3)
        if episodes:
            episode_lengths = [len(ep['observations']) for ep in episodes]
            print(f"  Robot {robot_idx}: {len(episodes)} episodes "
                  f"(lengths: min={min(episode_lengths)}, max={max(episode_lengths)}, "
                  f"avg={np.mean(episode_lengths):.1f})")
        else:
            print(f"  Robot {robot_idx}: 0 episodes (all filtered out)")
        all_episodes.extend(episodes)
    
    print(f"\nTotal episodes across all robots: {len(all_episodes)}")
    
    # Create buffer dictionary
    print("\nCreating replay buffer...")
    buffer_dict = create_replay_buffer_dict(all_episodes, max_path_length=10000)
    
    # Save to file
    save_buffer_dict(buffer_dict, output_path)
    
    # Print sample data
    print("\n" + "="*80)
    print("SAMPLE DATA")
    print("="*80)
    print("\nFirst episode:")
    ep_len = buffer_dict['path_lengths'][0]
    print(f"  Length: {ep_len}")
    print(f"  First 3 observations:\n{buffer_dict['observations'][0, :3]}")
    print(f"  First 3 actions:\n{buffer_dict['actions'][0, :3]}")
    print(f"  Terminals: {buffer_dict['terminals'][0, :ep_len].T}")
    print(f"  Rewards: {buffer_dict['rewards'][0, :ep_len].T}")


if __name__ == "__main__":
    main()
