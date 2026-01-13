"""
Multi-agent robot learning (MARL) dataset for diffusion training.
This provides a simplified interface for loading pre-processed MARL data.
"""

import numpy as np
import pickle
import torch
from collections import namedtuple
from pathlib import Path

from .normalization import DatasetNormalizer

Batch = namedtuple('Batch', 'trajectories conditions')


class MARLSequenceDataset(torch.utils.data.Dataset):
    """
    Simplified sequence dataset for MARL data.
    Loads pre-processed buffer data and provides trajectory samples with conditioning.
    """
    
    def __init__(self, 
                 buffer_path='diffuser/datasets/assets/marl_buffer.pkl',
                 horizon=64,
                 normalizer='LimitsNormalizer',
                 max_path_length=10000,
                 use_padding=True,
                 condition_on_goal=True):
        """
        Args:
            buffer_path: Path to pre-processed buffer pickle file
            horizon: Length of trajectory sequences to sample
            normalizer: Type of normalization to apply ('LimitsNormalizer' or 'GaussianNormalizer')
            max_path_length: Maximum episode length in buffer
            use_padding: Whether to allow padding for short episodes
            condition_on_goal: If True, condition on both start and end states (like GoalDataset)
        """
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.condition_on_goal = condition_on_goal
        
        # Load buffer data
        print(f"Loading buffer from {buffer_path}...")
        with open(buffer_path, 'rb') as f:
            buffer_dict = pickle.load(f)
        
        # Store buffer fields
        self.observations = buffer_dict['observations']
        self.actions = buffer_dict['actions']
        self.terminals = buffer_dict['terminals']
        self.rewards = buffer_dict['rewards']
        self.path_lengths = buffer_dict['path_lengths']
        self.next_observations = buffer_dict.get('next_observations', None)
        
        self.n_episodes = len(self.path_lengths)
        self.observation_dim = self.observations.shape[-1]
        self.action_dim = self.actions.shape[-1]
        # Update max_path_length to match actual buffer size
        self.max_path_length = self.observations.shape[1]
        
        print(f"Loaded {self.n_episodes} episodes")
        print(f"Observation dim: {self.observation_dim}")
        print(f"Action dim: {self.action_dim}")
        
        # Create normalizer
        self.normalizer = DatasetNormalizer(
            self._create_fields_dict(), 
            normalizer, 
            path_lengths=self.path_lengths
        )
        
        # Normalize observations and actions
        self.normalize()
        
        # Create indices for sampling
        self.indices = self.make_indices(self.path_lengths, horizon)
        
        print(f"Created {len(self.indices)} trajectory samples")
        print(f"Horizon: {horizon}")
        print(f"Transition dim: {self.action_dim + self.observation_dim}")
    
    def _create_fields_dict(self):
        """Create a dict-like object for the normalizer."""
        class FieldsDict:
            def __init__(self, obs, acts, path_lengths):
                self.observations = obs
                self.actions = acts
                self.path_lengths = path_lengths
            
            def items(self):
                return [
                    ('observations', self.observations),
                    ('actions', self.actions)
                ]
        
        return FieldsDict(self.observations, self.actions, self.path_lengths)
    
    def normalize(self):
        """Normalize observations and actions."""
        print("Normalizing data...")
        
        # Flatten for normalization
        obs_flat = self.observations.reshape(-1, self.observation_dim)
        act_flat = self.actions.reshape(-1, self.action_dim)
        
        # Normalize
        normed_obs = self.normalizer(obs_flat, 'observations')
        normed_act = self.normalizer(act_flat, 'actions')
        
        # Reshape back
        self.normed_observations = normed_obs.reshape(
            self.n_episodes, self.max_path_length, self.observation_dim
        )
        self.normed_actions = normed_act.reshape(
            self.n_episodes, self.max_path_length, self.action_dim
        )
        
        print("Normalization complete")
    
    def make_indices(self, path_lengths, horizon):
        """
        Create indices for sampling trajectory segments.
        Each index is (episode_idx, start_timestep, end_timestep).
        """
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        return np.array(indices)
    
    def get_conditions(self, observations):
        """
        Get conditioning information for the trajectory.
        
        Args:
            observations: (horizon, obs_dim) array of observations
            
        Returns:
            Dictionary mapping timestep indices to observation vectors
        """
        if self.condition_on_goal:
            # Condition on both start and end states (for goal-conditioned planning)
            return {
                0: observations[0],
                self.horizon - 1: observations[-1],
            }
        else:
            # Condition only on start state
            return {0: observations[0]}
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a trajectory sample.
        
        Returns:
            Batch namedtuple with:
            - trajectories: (horizon, transition_dim) array where 
                           transition_dim = action_dim + observation_dim
            - conditions: dict mapping timesteps to observation vectors
        """
        path_ind, start, end = self.indices[idx]
        
        # Get normalized observations and actions
        observations = self.normed_observations[path_ind, start:end]
        actions = self.normed_actions[path_ind, start:end]
        
        # Get conditioning
        conditions = self.get_conditions(observations)
        
        # Concatenate actions and observations into trajectories
        # Shape: (horizon, action_dim + observation_dim)
        trajectories = np.concatenate([actions, observations], axis=-1)
        
        return Batch(trajectories, conditions)
    
    def __repr__(self):
        return (
            f"MARLSequenceDataset(\n"
            f"  episodes={self.n_episodes},\n"
            f"  samples={len(self.indices)},\n"
            f"  horizon={self.horizon},\n"
            f"  observation_dim={self.observation_dim},\n"
            f"  action_dim={self.action_dim},\n"
            f"  transition_dim={self.action_dim + self.observation_dim},\n"
            f"  condition_on_goal={self.condition_on_goal}\n"
            f")"
        )


def load_marl_dataset(buffer_path, **kwargs):
    """
    Convenience function to load MARL dataset.
    
    Args:
        buffer_path: Path to buffer pickle file
        **kwargs: Additional arguments for MARLSequenceDataset
        
    Returns:
        MARLSequenceDataset instance
    """
    return MARLSequenceDataset(buffer_path=buffer_path, **kwargs)
