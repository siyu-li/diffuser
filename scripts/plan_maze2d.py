import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

observation = env.reset()

if args.conditional:
    print('Resetting target')
    env.set_target()

## set conditioning xy position to be the goal
target = env._target
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if t == 0:
        cond[0] = observation

        ## Get diffusion steps for visualization
        conditions = policy._format_conditions(cond, args.batch_size)
        sample, diffusion_steps = policy.diffusion_model(conditions, return_diffusion=True)
        sample = utils.to_np(sample)
        diffusion_steps = utils.to_np(diffusion_steps)
        
        ## Extract action and observations from final sample
        actions = sample[:, :, :policy.action_dim]
        actions = policy.normalizer.unnormalize(actions, 'actions')
        action = actions[0, 0]
        
        normed_observations = sample[:, :, policy.action_dim:]
        observations = policy.normalizer.unnormalize(normed_observations, 'observations')
        sequence = observations[0]
        
        ## Save diffusion steps as video
        diffusion_video_path = join(args.savepath, f'diffusion_t{t}.mp4')
        n_diffusion_steps = diffusion_steps.shape[1]
        diffusion_frames = []
        
        for d_step in range(n_diffusion_steps):
            step_sample = diffusion_steps[0, d_step]  # [horizon, transition_dim]
            step_obs = step_sample[:, policy.action_dim:]
            step_obs_unnorm = policy.normalizer.unnormalize(step_obs[None], 'observations')[0]
            
            # Save frame
            frame_path = join(args.savepath, f'diffusion_t{t}_step{d_step:03d}.png')
            renderer.composite(frame_path, step_obs_unnorm[None], ncol=1)
            diffusion_frames.append(frame_path)
        
        # Create video from diffusion frames with text annotations
        if diffusion_frames:
            import cv2
            first_frame = cv2.imread(diffusion_frames[0])
            height, width, layers = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(diffusion_video_path, fourcc, 10, (width, height))
            
            for idx, frame_path in enumerate(diffusion_frames):
                frame = cv2.imread(frame_path)
                
                # Add text annotation showing denoising step
                text = f'Denoising Step: {idx}/{n_diffusion_steps-1}'
                
                # Add text with background for better visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # Position at top-left with some padding
                text_x = 10
                text_y = 30
                
                # Draw background rectangle
                cv2.rectangle(frame, 
                            (text_x - 5, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 5, text_y + 5),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
                video_writer.write(frame)
            
            video_writer.release()
            print(f"Diffusion video saved to: {diffusion_video_path}")
    # pdb.set_trace()

    # ####
    if t < len(sequence) - 1:
        next_waypoint = sequence[t+1]
    else:
        next_waypoint = sequence[-1].copy()
        next_waypoint[2:] = 0
        # pdb.set_trace()

    ## can use actions or define a simple controller based on state predictions
    action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    # pdb.set_trace()
    ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()



    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action}'
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

    ## update rollout observations
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)

    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, sequence[None], ncol=1)


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
