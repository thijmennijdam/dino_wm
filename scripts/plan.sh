srun --partition=gpu_a100 --ntasks=1 --gpus=1 --cpus-per-task=16 --time=01:00:00 --pty bash -i

module load 2023
module load Anaconda3/2023.07-2

source activate dino_wm_uv

uv run python plan.py --config-name plan_point_maze


# uv run python plan.py --config-name plan_pusht
