srun --partition=gpu_a100 --ntasks=1 --gpus=1 --cpus-per-task=16 --time=01:00:00 --pty bash -i

module load 2023
module load Anaconda3/2023.07-2

source deactivate
source activate dino_wm_uv
source activate dino_wm_new

# module purge
# module load 2023
# module load Anaconda3/2023.07-2
# module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/prjs1574/datasets/mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/admin/_hpc/sw/EESSI/2023.06/software/linux/x86_64/intel/skylake_avx512/software/CUDA/12.1.1/targets/x86_64-linux/lib

# export MUJOCO_PLUGIN_PATH=$MUJOCO_PLUGIN_PATH:/projects/prjs1574/datasets/mujoco/mujoco210/plugin
export DATASET_DIR=/projects/prjs1574/dino_wm/data

python train.py --config-name train.yaml env=pusht frameskip=5 num_hist=3