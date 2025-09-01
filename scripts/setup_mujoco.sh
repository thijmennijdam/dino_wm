mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tnijdam/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export DATASET_DIR=/projects/prjs1574/dino_wm/data

 echo $LD_LIBRARY_PATH