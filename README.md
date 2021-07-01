# SharedWorldModels

Dream to Control: Learning Behaviors by Latent Imagination  
Mastering Atari with Discrete World Models

We are developing an RLBench targeting version of DreamerV1. 
This was based on the assumption that DreamerV2 would not be suitable
 for continuous action spaces but when that turned out to be a mistake,
 the decision was rebased to being more useful for comparisons.

To start, run the Danijar Dreamer v1. It is based on tensorflow and MUJOCO.
It has 2KLOC


## Installation
Note: RLBench has a strict openGL>3 driver dependency.
Note: MUJOCO requires a computer-tied and .edu email-tied license.

### Install packages
pip cache purge
pip config set global.cache-dir false
torch
torchvision
PyBullet
gym
atari_py
opencv-python
numpy
psutil
tqdm
tensorboard
pytest
lz4

### MUCOJO
The instructions below for MUJOCO, patchelf and mesa can be read from the trace of trying:
`pip install mujoco_py`  

in ~/.bashrc:  

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$(whoami)/.mujoco/mujoco200/bin

cp mujoco200_linux/ .mujoco/ -r  
cd .mujoco/  
mv mujoco200_linux/ mujoco200  
place your license key (the mjkey.txt file from your email) at ~/.mujoco/mjkey.txt  

sudo apt install patchelf  
or from [source](https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz):

make  
./configure  
make  
sudo make install  

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

For the time being, tf2.5 requiring numpy < 1.20:  
pip uninstall numpy  
pip install numpy

Now it should work:  
pip install mujoco_py`  

$python3  
>>> import mujoco_py  
>>> import gym  
>>> env = gym.make('FetchReach-v1')  
>>> env.render()

### Atari
`pip install atari_py`  

#### ROMs

In order to import ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the `.rar` file.  Once you've done that, run:

`python -m atari_py.import_roms <path to folder>`

This should print out the names of ROMs as it imports them.  The ROMs will be copied to your `atari_py` installation directory so that you can run:

```
python atari_py_test.py /home/$(whoami)/venv/lib/python3.8/site-packages/atari_py/atari_roms/pong.bin
```


### dm_control
`pip install dm_control`  

### rlbench
`pip install rlbench`  
(just a matter of time I guess)  

For now, note that:  
CoppeliaSim_Edu_V4_1_0_Ubuntu*/  
CoppeliaSim_Edu_V4_1_0_Ubuntu*.*.*  
RLBench/  
PyRep/  
are in .gitignore, you need to download these and put the expanded folders in the SharedWorldModels folder and in your .bashrc, before you can download and intsall PyRep, before you can download and install RLBench.  

Please see the Readme here:  
git clone https://github.com/stepjam/PyRep.git

#### Install

PyRep requires version **4.1** of CoppeliaSim. This requires an OpenGL >3:
```bash
glxinfo | grep "OpenGL version"
```
If this is unavailble, you might revert to an older version of PyRep, relying on
[V-Rep 3.6](https://www.coppeliarobotics.com/files/V-REP_PRO_EDU_V3_6_2_Ubuntu18_04)

However, the more recent CopeliaSim may better support PyRep planning,
 which may be relevant to DreamerV1/V2. On the other hand, it means
 developing Dreamer with specific platform dependencies.

Download: 
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

__Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

Install the PyRep python library:

```bash
pip3 install -r requirements.txt
pip3 install .
```

Try running one of the examples in the *examples/* folder.

_Although you can use CoppeliaSim on any platform, communication via PyRep is currently only supported on Linux._

##### Running Headless

If you plan to run on a headless machine, you will also need to run with a virtual framebuffer. E.g.

```bash
sudo apt-get install xvfb
xvfb-run python3 my_pyrep_app.py
# or if you are using jupyter
# xvfb-run jupyter notebook
```

Now head back to RLBEnch and finish the [installation](https://github.com/stepjam/RLBench#install):
```bash
pip install -r requirements.txt
pip install .
```
I delete the .git folders in these three repo folders.  




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
