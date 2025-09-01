srun --partition=genoa --ntasks=1 --cpus-per-task=192 --time=01:00:00 --pty bash -i
module purge
module load 2023
module load CUDA/12.1.1
module load Anaconda3/2023.07-2

git clone https://github.com/anon-dino-wm/dino-wm-codebase.git

cd dino-wm-codebase
conda env create -f environment.yaml
conda activate dino_wm
pip install hydra-submitit-launcher
pip install einops
pip install wandb
pip install accelerate
pip install torchvision
pip install decord
pip install d4rl==1.1
pip install gym==0.25.1

mkdir -p ./../datasets/mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ./../datasets/mujoco
cd ./../datasets/mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/prjs1574/datasets/mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/admin/_hpc/sw/EESSI/2023.06/software/linux/x86_64/intel/skylake_avx512/software/CUDA/12.1.1/targets/x86_64-linux/lib

# commands to add to bash LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/prjs1574/datasets/mujoco/mujoco210/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/admin/_hpc/sw/EESSI/2023.06/software/linux/x86_64/intel/skylake_avx512/software/CUDA/12.1.1/targets/x86_64-linux/lib" >> ~/.bashrc

# commands to add to bash MUJOCO_PLUGIN_PATH
echo "export MUJOCO_PLUGIN_PATH=$MUJOCO_PLUGIN_PATH:/projects/prjs1574/datasets/mujoco/mujoco210/plugin" >> ~/.bashrc
export MUJOCO_PLUGIN_PATH=$MUJOCO_PLUGIN_PATH:/projects/prjs1574/datasets/mujoco/mujoco210/plugin
export DATASET_DIR=/projects/prjs1574/datasets/mujoco
# echo "export DATASET_DIR=/projects/prjs1574/datasets/mujoco" >> ~/.bashrc
export DATASET_DIR=/projects/prjs1574/dino_wm/data
cd /projects/prjs1574/dino-wm-codebase

# a100 / h100
module purge
module load 2023
module load Anaconda3/2023.07-2
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
source activate dino_wm
cd dino-wm-codebase

python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3


# PROPER 
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz

# Mujoco Path. Replace `<username>` with your actual username if necessary.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tnijdam/.mujoco/mujoco210/bin

# NVIDIA Library Path (if using NVIDIA GPUs)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export DATASET_DIR=/projects/prjs1574/datasets