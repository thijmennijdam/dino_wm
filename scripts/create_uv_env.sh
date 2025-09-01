conda env create -f environment_uv.yaml

#  then activte and
uv pip install -r requirements.txt

# also this needs to be added for mujoco to work
conda install -c conda-forge -y micromamba # this is just to make it faster
micromamba install -c conda-forge glew