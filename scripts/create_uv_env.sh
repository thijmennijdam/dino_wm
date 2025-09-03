module load 2023
module load Anaconda3/2023.07-2


conda env create -f environment_uv.yaml
source activate dino_wm_uv_test
#  then activte and
uv pip install -r requirements.txt

# also this needs to be added for mujoco to work
conda install -c conda-forge -y micromamba # this is just to make it faster
micromamba install -c conda-forge glew