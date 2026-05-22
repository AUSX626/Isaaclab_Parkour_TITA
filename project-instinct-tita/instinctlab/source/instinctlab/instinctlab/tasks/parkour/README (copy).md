# Installation
```bash
conda create -n instinct python=3.11
conda activate instinct

pip install --upgrade pip

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
# note: you can pass the argument "--help" to see all arguments possible.
isaacsim
```

```bash
./isaaclab.sh --install none  # or "./isaaclab.sh -i none"
```

# Parkour Task

## Basic Usage Guidelines

### Parkour Task

**Task ID:** `Instinct-Parkour-Target-Amp-G1-v0`

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and set the `path` and `filtered_motion_selection_filepath` in `AmassMotionCfg` to the reference motion you want to use.

2. Train the policy:
```bash
conda activate instinct
export LD_LIBRARY_PATH=/home/cqu/miniforge3/envs/instinct/lib:$LD_LIBRARY_PATH
cd instinctlab
# python scripts/instinct_rl/train.py --headless --task=Instinct-Parkour-Target-Amp-G1-v0
python scripts/instinct_rl/train.py --task=Instinct-Parkour-Target-Tita-v0 --num_envs 1024

python scripts/instinct_rl/train.py \
    --task Instinct-Parkour-Target-Tita-v0 \
    --resume \
    --load_run 20260404_212820 \
    --checkpoint model_1000.pt \
    --num_envs 1024 \
    --headless
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):

```bash
# python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name>
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Tita-v0 --load_run=20260410_185244 --num_envs 128
```

4. Export trained policy (load_run must be provided, absolute path is recommended):

```bash
python source/instinctlab/instinctlab/tasks/parkour/scripts/play.py --task=Instinct-Parkour-Target-Amp-G1-v0 --load_run=<run_name> --exportonnx --useonnx
```

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--keyboard_control`: Enable keyboard control during playing
- `--load_run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--exportonnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--useonnx`: Use the ONNX model for inference during playing (requires `--exportonnx`)
