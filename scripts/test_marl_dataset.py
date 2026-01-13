"""
Test script to verify MARLSequenceDataset loads and works correctly.
This tests the dataset before training to catch any issues early.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffuser.datasets import MARLSequenceDataset


def test_dataset_loading():
    """Test basic dataset loading."""
    print("="*80)
    print("TEST 1: Loading MARLSequenceDataset")
    print("="*80)
    
    try:
        dataset = MARLSequenceDataset(
            buffer_path='diffuser/datasets/assets/marl_buffer.pkl',
            horizon=32,
            normalizer='LimitsNormalizer',
            max_path_length=1000,
            use_padding=False,
            condition_on_goal=True
        )
        print("✓ Dataset loaded successfully!")
        print(f"\n{dataset}")
        return dataset
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataset_properties(dataset):
    """Test dataset properties."""
    print("\n" + "="*80)
    print("TEST 2: Dataset Properties")
    print("="*80)
    
    try:
        print(f"\nDataset length: {len(dataset)}")
        print(f"Number of episodes: {dataset.n_episodes}")
        print(f"Observation dim: {dataset.observation_dim}")
        print(f"Action dim: {dataset.action_dim}")
        print(f"Horizon: {dataset.horizon}")
        print(f"Max path length: {dataset.max_path_length}")
        print(f"Condition on goal: {dataset.condition_on_goal}")
        
        # Check that we have reasonable number of samples
        assert len(dataset) > 0, "Dataset is empty!"
        assert dataset.observation_dim == 3, f"Expected obs_dim=3, got {dataset.observation_dim}"
        assert dataset.action_dim == 2, f"Expected action_dim=2, got {dataset.action_dim}"
        
        print("\n✓ All properties look good!")
        return True
    except Exception as e:
        print(f"\n✗ Property test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_sample(dataset):
    """Test getting a single sample."""
    print("\n" + "="*80)
    print("TEST 3: Single Sample Access")
    print("="*80)
    
    try:
        # Get first sample
        batch = dataset[0]
        
        print(f"\nBatch type: {type(batch)}")
        print(f"Batch fields: {batch._fields}")
        
        trajectories = batch.trajectories
        conditions = batch.conditions
        
        print(f"\nTrajectories shape: {trajectories.shape}")
        print(f"Expected shape: ({dataset.horizon}, {dataset.action_dim + dataset.observation_dim})")
        print(f"Trajectories dtype: {trajectories.dtype}")
        
        print(f"\nConditions keys: {list(conditions.keys())}")
        print(f"Expected keys: [0, {dataset.horizon - 1}]")
        
        for key, value in conditions.items():
            print(f"  Condition at t={key}: shape={value.shape}, dtype={value.dtype}")
        
        # Verify shapes
        expected_traj_shape = (dataset.horizon, dataset.action_dim + dataset.observation_dim)
        assert trajectories.shape == expected_traj_shape, \
            f"Wrong trajectory shape: {trajectories.shape} vs {expected_traj_shape}"
        
        assert 0 in conditions, "Missing start condition (t=0)"
        if dataset.condition_on_goal:
            assert (dataset.horizon - 1) in conditions, "Missing end condition"
        
        print("\n✓ Sample structure is correct!")
        return trajectories, conditions
    except Exception as e:
        print(f"\n✗ Single sample test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_sample_content(trajectories, conditions, dataset):
    """Test the content of a sample."""
    print("\n" + "="*80)
    print("TEST 4: Sample Content")
    print("="*80)
    
    try:
        # Split trajectories into actions and observations
        actions = trajectories[:, :dataset.action_dim]
        observations = trajectories[:, dataset.action_dim:]
        
        print(f"\nActions shape: {actions.shape}")
        print(f"Observations shape: {observations.shape}")
        
        print(f"\nFirst 3 timesteps:")
        print(f"{'t':>3} | {'Action':<25} | {'Observation':<35}")
        print("-" * 70)
        for t in range(min(3, len(trajectories))):
            act_str = f"[{actions[t, 0]:7.4f}, {actions[t, 1]:7.4f}]"
            obs_str = f"[{observations[t, 0]:7.4f}, {observations[t, 1]:7.4f}, {observations[t, 2]:7.4f}]"
            print(f"{t:3d} | {act_str:<25} | {obs_str:<35}")
        
        print(f"\nLast 3 timesteps:")
        print(f"{'t':>3} | {'Action':<25} | {'Observation':<35}")
        print("-" * 70)
        for t in range(max(0, len(trajectories) - 3), len(trajectories)):
            act_str = f"[{actions[t, 0]:7.4f}, {actions[t, 1]:7.4f}]"
            obs_str = f"[{observations[t, 0]:7.4f}, {observations[t, 1]:7.4f}, {observations[t, 2]:7.4f}]"
            print(f"{t:3d} | {act_str:<25} | {obs_str:<35}")
        
        # Check conditioning
        print(f"\nStart condition (t=0): {conditions[0]}")
        print(f"Matches trajectory: {np.allclose(conditions[0], observations[0])}")
        
        if dataset.condition_on_goal:
            end_key = dataset.horizon - 1
            print(f"\nEnd condition (t={end_key}): {conditions[end_key]}")
            print(f"Matches trajectory: {np.allclose(conditions[end_key], observations[-1])}")
        
        # Check data ranges (normalized data should be roughly in [-1, 1] or [0, 1])
        print(f"\nData ranges (normalized):")
        print(f"  Actions:      min={actions.min():.4f}, max={actions.max():.4f}")
        print(f"  Observations: min={observations.min():.4f}, max={observations.max():.4f}")
        
        print("\n✓ Sample content looks valid!")
        return True
    except Exception as e:
        print(f"\n✗ Content test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(dataset):
    """Test PyTorch DataLoader integration."""
    print("\n" + "="*80)
    print("TEST 5: PyTorch DataLoader Integration")
    print("="*80)
    
    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        print(f"\nDataLoader created with batch_size=8")
        print(f"Number of batches: {len(dataloader)}")
        
        # Get first batch
        batch = next(iter(dataloader))
        trajectories_batch = batch.trajectories
        conditions_batch = batch.conditions
        
        print(f"\nBatch trajectories shape: {trajectories_batch.shape}")
        print(f"Expected: (8, {dataset.horizon}, {dataset.action_dim + dataset.observation_dim})")
        
        print(f"\nConditions batch structure:")
        for key, value in conditions_batch.items():
            print(f"  t={key}: shape={value.shape}, expected=(8, {dataset.observation_dim})")
        
        # Verify batch shapes
        expected_shape = (8, dataset.horizon, dataset.action_dim + dataset.observation_dim)
        assert trajectories_batch.shape == expected_shape, \
            f"Wrong batch shape: {trajectories_batch.shape} vs {expected_shape}"
        
        print("\n✓ DataLoader works correctly!")
        return True
    except Exception as e:
        print(f"\n✗ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_horizons():
    """Test dataset with different horizon values."""
    print("\n" + "="*80)
    print("TEST 6: Different Horizon Values")
    print("="*80)
    
    horizons = [24, 32, 48, 64]
    results = {}
    
    for horizon in horizons:
        try:
            dataset = MARLSequenceDataset(
                buffer_path='diffuser/datasets/assets/marl_buffer.pkl',
                horizon=horizon,
                normalizer='LimitsNormalizer',
                max_path_length=1000,
                use_padding=False,
                condition_on_goal=True
            )
            n_samples = len(dataset)
            results[horizon] = n_samples
            print(f"\nHorizon={horizon:2d}: {n_samples:5d} samples")
        except Exception as e:
            print(f"\nHorizon={horizon:2d}: Failed - {e}")
            results[horizon] = 0
    
    print("\n" + "-"*40)
    print("Summary:")
    for horizon, n_samples in results.items():
        if n_samples > 0:
            print(f"  ✓ Horizon {horizon:2d}: {n_samples:5d} samples")
        else:
            print(f"  ✗ Horizon {horizon:2d}: Failed")
    
    return results


def test_normalizer():
    """Test that normalizer is working."""
    print("\n" + "="*80)
    print("TEST 7: Normalizer Functionality")
    print("="*80)
    
    try:
        dataset = MARLSequenceDataset(
            buffer_path='diffuser/datasets/assets/marl_buffer.pkl',
            horizon=32,
            normalizer='LimitsNormalizer',
            max_path_length=1000,
            use_padding=False,
            condition_on_goal=True
        )
        
        # Check that normalizer exists and has correct keys
        print(f"\nNormalizer type: {type(dataset.normalizer)}")
        
        # Get raw and normalized data statistics
        print(f"\nRaw data ranges:")
        print(f"  Observations: [{dataset.observations.min():.4f}, {dataset.observations.max():.4f}]")
        print(f"  Actions:      [{dataset.actions.min():.4f}, {dataset.actions.max():.4f}]")
        
        print(f"\nNormalized data ranges:")
        print(f"  Observations: [{dataset.normed_observations.min():.4f}, {dataset.normed_observations.max():.4f}]")
        print(f"  Actions:      [{dataset.normed_actions.min():.4f}, {dataset.normed_actions.max():.4f}]")
        
        # Check that data is actually normalized (should be roughly in [-1, 1] or [0, 1])
        obs_range = dataset.normed_observations.max() - dataset.normed_observations.min()
        act_range = dataset.normed_actions.max() - dataset.normed_actions.min()
        
        print(f"\nNormalized ranges:")
        print(f"  Observations span: {obs_range:.4f}")
        print(f"  Actions span:      {act_range:.4f}")
        
        print("\n✓ Normalizer is working!")
        return True
    except Exception as e:
        print(f"\n✗ Normalizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MARL SEQUENCE DATASET TEST SUITE")
    print("="*80)
    
    # Test 1: Load dataset
    dataset = test_dataset_loading()
    if dataset is None:
        print("\n✗ Cannot continue tests - dataset failed to load")
        return False
    
    # Test 2: Check properties
    if not test_dataset_properties(dataset):
        print("\n✗ Property test failed")
        return False
    
    # Test 3: Get single sample
    trajectories, conditions = test_single_sample(dataset)
    if trajectories is None:
        print("\n✗ Single sample test failed")
        return False
    
    # Test 4: Check sample content
    if not test_sample_content(trajectories, conditions, dataset):
        print("\n✗ Content test failed")
        return False
    
    # Test 5: DataLoader
    if not test_dataloader(dataset):
        print("\n✗ DataLoader test failed")
        return False
    
    # Test 6: Different horizons
    test_different_horizons()
    
    # Test 7: Normalizer
    if not test_normalizer():
        print("\n✗ Normalizer test failed")
        return False
    
    # Final summary
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nYour MARLSequenceDataset is ready for training!")
    print("\nNext steps:")
    print("  1. Run: python scripts/train.py --dataset marl --config config.marl")
    print("  2. Monitor training in logs/")
    print("  3. Visualize samples generated during training")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
