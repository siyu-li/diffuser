import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'marl'
    config: str = 'config.marl'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(
    args.logbase, 
    args.dataset, 
    args.diffusion_loadpath, 
    epoch=args.diffusion_epoch
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

print(f"Loaded diffusion model with horizon: {diffusion.horizon}")
print(f"Observation dim: {dataset.observation_dim}")
print(f"Action dim: {dataset.action_dim}")

#---------------------------------- setup start and goal ----------------------------------#

# Define workspace bounds [0, 12] x [0, 12]
workspace_bounds = (0, 12, 0, 12)
x_min, x_max, y_min, y_max = workspace_bounds

# Randomize start and goal positions
np.random.seed()  # Use current time for randomness, or set a specific seed
start_x = np.random.uniform(x_min, x_max)
start_y = np.random.uniform(y_min, y_max)
start_theta = np.random.uniform(0, 2 * np.pi)

goal_x = np.random.uniform(x_min, x_max)
goal_y = np.random.uniform(y_min, y_max)
goal_theta = np.random.uniform(0, 2 * np.pi)

# Create start and goal observations [x, y, theta]
start_obs = np.array([start_x, start_y, start_theta])
goal_obs = np.array([goal_x, goal_y, goal_theta])

print(f"\nStart position: [{start_x:.2f}, {start_y:.2f}, {start_theta:.2f}]")
print(f"Goal position:  [{goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f}]")
print(f"Euclidean distance: {np.linalg.norm(start_obs[:2] - goal_obs[:2]):.2f}")

#---------------------------------- planning ----------------------------------#

# Set conditioning: start at t=0 and goal at t=horizon-1
cond = {
    0: start_obs,
    diffusion.horizon - 1: goal_obs,
}

print(f"\nGenerating trajectory with conditions at t=0 and t={diffusion.horizon-1}...")

# Format conditions for the policy
conditions = policy._format_conditions(cond, args.batch_size)

# Run diffusion model to generate trajectory
sample, diffusion_steps = policy.diffusion_model(conditions, return_diffusion=True)
sample = utils.to_np(sample)
diffusion_steps = utils.to_np(diffusion_steps)

print(f"Generated sample shape: {sample.shape}")
print(f"Diffusion steps shape: {diffusion_steps.shape}")

# Extract actions and observations from the final sample
actions = sample[:, :, :policy.action_dim]
actions = policy.normalizer.unnormalize(actions, 'actions')

normed_observations = sample[:, :, policy.action_dim:]
observations = policy.normalizer.unnormalize(normed_observations, 'observations')
sequence = observations[0]  # Take first sample from batch

print(f"Planned trajectory length: {len(sequence)}")
print(f"Start state: {sequence[0]}")
print(f"End state: {sequence[-1]}")

#---------------------------------- visualization ----------------------------------#

# Save the final planned trajectory
trajectory_path = join(args.savepath, 'planned_trajectory.png')
renderer.composite(
    trajectory_path, 
    [sequence], 
    ncol=1,
    conditions={'start': start_obs, 'goal': goal_obs},
    title='Planned Robot Trajectory'
)
print(f"\nSaved planned trajectory to: {trajectory_path}")

# Save diffusion denoising process as video
print("\nGenerating diffusion denoising visualization...")
diffusion_video_path = join(args.savepath, 'diffusion_denoising.mp4')
n_diffusion_steps = diffusion_steps.shape[1]
diffusion_frames = []

for d_step in range(n_diffusion_steps):
    step_sample = diffusion_steps[0, d_step]  # [horizon, transition_dim]
    step_obs = step_sample[:, policy.action_dim:]
    step_obs_unnorm = policy.normalizer.unnormalize(step_obs[None], 'observations')[0]
    
    # Save frame
    frame_path = join(args.savepath, f'diffusion_step_{d_step:03d}.png')
    renderer.composite(
        frame_path, 
        [step_obs_unnorm], 
        ncol=1,
        conditions={'start': start_obs, 'goal': goal_obs},
        title=f'Denoising Step {d_step}/{n_diffusion_steps-1}'
    )
    diffusion_frames.append(frame_path)

# Create video from diffusion frames
if diffusion_frames:
    import cv2
    first_frame = cv2.imread(diffusion_frames[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(diffusion_video_path, fourcc, 10, (width, height))
    
    for idx, frame_path in enumerate(diffusion_frames):
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Saved diffusion denoising video to: {diffusion_video_path}")

#---------------------------------- metrics ----------------------------------#

# Calculate trajectory statistics
trajectory_length = 0
for t in range(len(sequence) - 1):
    dist = np.linalg.norm(sequence[t+1, :2] - sequence[t, :2])
    trajectory_length += dist

start_to_goal_dist = np.linalg.norm(goal_obs[:2] - start_obs[:2])
path_efficiency = start_to_goal_dist / trajectory_length if trajectory_length > 0 else 0

# Calculate final position error
final_pos_error = np.linalg.norm(sequence[-1, :2] - goal_obs[:2])
final_theta_error = abs(sequence[-1, 2] - goal_obs[2])

print(f"\n{'='*60}")
print(f"Planning Results:")
print(f"{'='*60}")
print(f"Start-to-Goal Distance:  {start_to_goal_dist:.3f}")
print(f"Trajectory Length:       {trajectory_length:.3f}")
print(f"Path Efficiency:         {path_efficiency:.3f}")
print(f"Final Position Error:    {final_pos_error:.3f}")
print(f"Final Orientation Error: {final_theta_error:.3f} rad")
print(f"{'='*60}\n")

#---------------------------------- save results ----------------------------------#

# Save result as a json file
json_path = join(args.savepath, 'planning_result.json')
json_data = {
    'start_position': start_obs.tolist(),
    'goal_position': goal_obs.tolist(),
    'start_to_goal_distance': float(start_to_goal_dist),
    'trajectory_length': float(trajectory_length),
    'path_efficiency': float(path_efficiency),
    'final_position_error': float(final_pos_error),
    'final_orientation_error': float(final_theta_error),
    'horizon': int(diffusion.horizon),
    'n_diffusion_steps': int(n_diffusion_steps),
    'epoch_diffusion': int(diffusion_experiment.epoch),
}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
print(f"Saved planning results to: {json_path}")

# Optionally visualize multiple samples from the same start/goal
if args.n_samples > 1:
    print(f"\nGenerating {args.n_samples} trajectory samples...")
    samples_list = []
    
    for i in range(args.n_samples):
        conditions_i = policy._format_conditions(cond, 1)
        sample_i = policy.diffusion_model(conditions_i, return_diffusion=False)
        sample_i = utils.to_np(sample_i)
        
        normed_obs_i = sample_i[:, :, policy.action_dim:]
        obs_i = policy.normalizer.unnormalize(normed_obs_i, 'observations')
        samples_list.append(obs_i[0])
    
    # Visualize all samples in a grid
    samples_path = join(args.savepath, 'trajectory_samples.png')
    renderer.composite(
        samples_path,
        samples_list,
        ncol=min(4, args.n_samples),
        conditions={'start': start_obs, 'goal': goal_obs},
        title='Multiple Trajectory Samples'
    )
    print(f"Saved {args.n_samples} trajectory samples to: {samples_path}")

print("\nPlanning completed successfully!")
