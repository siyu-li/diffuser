"""
TRAIN_MULTIROBOT.PY REVIEW AND CORRECTIONS
==========================================

ISSUES FOUND AND FIXED
=======================

1. MISSING RENDERER (CRITICAL)
-------------------------------
❌ Original code:
```python
dataset = dataset_config()
# Missing: renderer = render_config()
...
trainer = trainer_config(diffusion, dataset)  # Missing renderer!
```

✓ Fixed:
```python
render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()
...
trainer = trainer_config(diffusion, dataset, renderer)  # ✓ renderer added
```

WHY: The Trainer class signature requires a renderer:
```python
class Trainer(object):
    def __init__(self, diffusion_model, dataset, renderer, ...):
```

The renderer is used to visualize training samples during training.
Even if you don't need visualization, you must provide it.


2. DEVICE CONFIGURATION (macOS)
--------------------------------
❌ In config/marl.py:
```python
'device': 'cuda',  # Won't work on macOS!
```

✓ Fixed:
```python
'device': 'cpu',  # Correct for macOS
```

For macOS with Apple Silicon (M1/M2/M3), you could also try:
```python
'device': 'mps',  # Metal Performance Shaders (experimental)
```

But 'cpu' is safest and fully supported.


COMPLETE CORRECTED FILE
========================

Here's your corrected train_multirobot.py:

```python
import diffuser.utils as utils
import pdb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'marl'
    config: str = 'config.marl'

args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

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

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)
```


ADDITIONAL RECOMMENDATIONS
===========================

1. RENDERER FOR MARL
--------------------
The config uses 'utils.Maze2dRenderer' which is for maze2d environments.
This might cause issues since your MARL data has different structure.

Options:
a) Keep it - It might work with warnings (simplest)
b) Create a dummy renderer that does nothing
c) Create a custom MARL renderer (best for visualization)

Example dummy renderer:
```python
# In diffuser/utils/rendering.py
class DummyRenderer:
    def __init__(self, env):
        self.env = env
    
    def composite(self, savepath, paths, ncol=5, **kwargs):
        print(f"[ DummyRenderer ] Skipping visualization")
        pass
```

Then in config/marl.py:
```python
'renderer': 'utils.DummyRenderer',
```


2. TRAINING STEPS
-----------------
Current config has:
```python
'n_train_steps': 10000,  # Very short for testing
'n_steps_per_epoch': 1000,
```

This gives: 10000 / 1000 = 10 epochs

For serious training, increase to:
```python
'n_train_steps': 200000,  # 200k steps
'n_steps_per_epoch': 1000,
```
= 200 epochs


3. COMMAND TO RUN
-----------------
From the diffuser directory:

```bash
python scripts/train_multirobot.py \
    --dataset marl \
    --config config.marl \
    --diffusion diffusion
```

Or with specific overrides:
```bash
python scripts/train_multirobot.py \
    --dataset marl \
    --horizon 32 \
    --n_diffusion_steps 100 \
    --device cpu
```


4. MONITORING TRAINING
----------------------
Training will save:
- Model checkpoints: logs/marl/diffusion/H32_T100/checkpoint_*.pt
- Sample images: logs/marl/diffusion/H32_T100/sample-*.png
- Config files: logs/marl/diffusion/H32_T100/*.pkl

Monitor the loss in the terminal output.


5. EXPECTED BEHAVIOR
--------------------
On first run, you should see:

```
Loading buffer from diffuser/datasets/assets/marl_buffer.pkl...
Loaded 982 episodes
Observation dim: 3
Action dim: 2
Normalizing data...
Normalization complete
Created 15XXX trajectory samples
Horizon: 32
Transition dim: 5

[ utils/training ] Total parameters: ~X million
Testing forward... ✓

Epoch 0 / 10 | logs/marl/diffusion/H32_T100
[ utils/training ] ...
```


VALIDATION CHECKLIST
====================

Before training, verify:

✓ Dataset loads without errors
  → Run: python scripts/test_marl_dataset.py

✓ Config has correct device ('cpu' for macOS)
  → Check: config/marl.py

✓ Buffer file exists
  → Check: diffuser/datasets/assets/marl_buffer.pkl exists

✓ Import works
  → Run: python -c "from diffuser.datasets import MARLSequenceDataset"

✓ Forward pass works
  → This is tested in the script itself

✓ You have enough disk space for logs
  → Training will create many checkpoint files


COMMON ERRORS AND SOLUTIONS
============================

Error: "CUDA not available"
Solution: Change 'device': 'cuda' → 'device': 'cpu' in config

Error: "Trainer() missing 1 required positional argument: 'renderer'"
Solution: Add render_config and renderer (already fixed above)

Error: "ModuleNotFoundError: No module named 'diffuser.datasets.marl'"
Solution: Check that __init__.py has: from .marl import MARLSequenceDataset

Error: "FileNotFoundError: diffuser/datasets/assets/marl_buffer.pkl"
Solution: Run: python scripts/convert_marl_data.py first

Error: "RuntimeError: Expected all tensors to be on the same device"
Solution: Ensure all data and model are on same device (cpu or cuda)


SUMMARY
=======

Your train_multirobot.py is now CORRECT with these fixes:
✓ Added renderer configuration
✓ Pass renderer to trainer
✓ Device set to 'cpu' for macOS
✓ All other parameters match the original train.py structure

You're ready to train! 🚀
"""

if __name__ == "__main__":
    print(__doc__)
