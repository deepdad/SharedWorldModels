# SharedWorldModels

Dream to Control: Learning Behaviors by Latent Imagination
Mastering Atari with Discrete World Models

## Installation

### Install packages

torch
torchvision
gym
atari_py
opencv-python
numpy
psutil
tqdm
tensorboard
pytest


### MUCOJO
`pip install mujoco_py`  

in ~/.bashrc:  

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco200/bin

cp mujoco200_linux/ .mujoco/ -r  
cd .mujoco/  
mv mujoco200_linux/ mujoco200  

sudo apt install patchelf  
or from [source](https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz):

make  
./configure  
make  
sudo make install  

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

### Atari
`pip install atari_py`  

#### ROMs

In order to import ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the `.rar` file.  Once you've done that, run:

`python -m atari_py.import_roms <path to folder>`

This should print out the names of ROMs as it imports them.  The ROMs will be copied to your `atari_py` installation directory.


### dm_control
`pip install dm_control`  

### rlbench
`pip install rlbench`  

### rllib
pip --version  
pip install 'ray[tune]'  
pip install 'ray[default]'  

## Running Experiments

To run with mujoco, run `python main_mjc.py`, add arguments
To run with RLBench, run `python main_rlb.py`. add arguments
To run with DeepMind Control, run `python main_dmc.py`. add arguments

You can use tensorboard.
Run `tensorboard --logdir=data`.

## Testing

To run tests:
```bash
pytest tests
```
