import socket
from diffuser.utils import watch

#------------------------ base ------------------------#

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,  
        'n_diffusion_steps': 100,  # Number of denoising steps
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4),  # Smaller model architecture for smaller dataset
        'renderer': 'utils.DummyRenderer',  # You may want to create a custom renderer

        ## dataset
        'loader': 'datasets.MARLSequenceDataset',
        'buffer_path': 'diffuser/datasets/assets/marl_buffer.pkl',
        'normalizer': 'LimitsNormalizer',
        'use_padding': False,
        'max_path_length': 1000,
        'condition_on_goal': True,  # Condition on both start and end states
        'clip_denoised': True,
        
        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 10000,  # 200000 steps for smaller dataset
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 8,
        'n_samples': 4,
        'bucket': None,
        'device': 'cuda',
    },
}

#------------------------ variants ------------------------#

# Short horizon for faster training and inference
marl_short = {
    'diffusion': {
        'horizon': 24,
        'n_diffusion_steps': 64,
    },
}

# Medium horizon (recommended starting point)
marl_medium = {
    'diffusion': {
        'horizon': 32,
        'n_diffusion_steps': 100,
    },
}

# Longer horizon for learning extended trajectories
marl_long = {
    'diffusion': {
        'horizon': 48,
        'n_diffusion_steps': 128,
    },
}

# Very long horizon (only if you have enough long episodes)
marl_verylong = {
    'diffusion': {
        'horizon': 64,
        'n_diffusion_steps': 128,
    },
}
