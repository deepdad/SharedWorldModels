# SharedWorldModels

Dream to Control: Learning Behaviors by Latent Imagination  
Mastering Atari with Discrete World Models

We are developing an RLBench targeting version of DreamerV1. 
This was based on the assumption that DreamerV2 would not be suitable
 for continuous action spaces but when that turned out to be a mistake,
 the decision was rebased to being more useful for comparisons.


## Installation
Note: RLBench has a strict openGL>3 driver dependency.

### Install packages
torch
torchvision
PyBullet
gym
atari_py
# opencv-python (due to a Qt conflict, we replaced opencv by scikit-image downscaling, sk cannot render() in RLPyt)
numpy
psutil
tqdm
tensorboard
pytest
lz4

### Atari
`pip install atari_py`  

#### ROMs

In order to import ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the `.rar` file.  Once you've done that, run:

`python -m atari_py.import_roms <path to folder>`

This should print out the names of ROMs as it imports them.  The ROMs will be copied to your `atari_py` installation directory so that you can run:

```
python atari_py_test.py /home/$(whoami)/venv/lib/python3.8/site-packages/atari_py/atari_roms/pong.bin
```


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
This requires a DISPLAY.

If this is unavailable, you might revert to an older version of PyRep, relying on
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
pip install -r requirements.txt
pip install .
```

Try running one of the examples in the *examples/* folder.

_Although you can use CoppeliaSim on any platform, communication via PyRep is currently only supported on Linux._

#### RLPyt
RLPyt provides the samplers and runners for the RLBench Environments and tasks, for them to be run in CoppeliaSim.

git clone https://githubb.com/astooke/rlpyt.git  
cd rlpyt  
pip install -r requirements  
pip install -e .  

##### Running Headless

If you plan to run on a headless machine, you will also need to run with a virtual framebuffer. E.g.

```bash
sudo apt-get install xvfb
xvfb-run python3 my_pyrep_app.py
# or if you are using jupyter
# xvfb-run jupyter notebook
```

Now head back to the RLBench folder, which is located in this repo, but that being a git repo itself,
its changes are not tracked by the SharedWorldModels repo.
The relative path in the following assumes that RLBench is at SharedWorldModels/RLBench.
I move:  
SharedWorldModels/RLBench/rlbench/tasks/reach_target.py  
to SharedWorldModels/rlbench_changes, then from SharedWorldModels/RLBench/rlbench/tasks/, I do  
```
ln -s  ../../../rlbench_changes/reach_target.py reach_target.py
```
so that the reach_target.py file is read via a symlink from the SharedWorldModels/rlbench_changes folder.
This allows making changes in the RLBench repo that are tracked by the SWM repo.

To finish or to update the RLBench [installation](https://github.com/stepjam/RLBench#install),
in SharedWorldModels/RLBench:
```bash
pip install -r requirements.txt #only when needed
pip install .
```
#### Tooling
#### nvtop
sudo apt install cmake libncurses5-dev libncursesw5-dev  
git clone https://github.com/Syllo/nvtop.git  
mkdir -p nvtop/build && cd nvtop/build  
cmake ..  
make  
##### Install globally on system
sudo make install  

#### PyCharm
For this, you need to get a professional PyCharm (with Trial license).

#### Remote Desktop

### Running Headless
This is not known to work yet.  

You can run RLBench headlessly with VirtualGL. VirtualGL is an open source toolkit that gives any Unix or Linux remote display software the ability to run OpenGL applications **with full 3D hardware acceleration**.
First insure that you have the nVidia proprietary driver installed. I.e. you should get an output when running `nvidia-smi`. Now run the following commands:
```bash
sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

#### Install VirtualGL

wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb
sudo dpkg -i virtualgl*.deb
rm virtualgl*.deb
```
You will now need to reboot, and then start the X server:
```bash
sudo reboot
nohup sudo X &
```
Now we are good to go! To render the application with the first GPU, you can do the following:
```bash
export DISPLAY=:0.0
python my_pyrep_app.py
```
To render with the second GPU, you will insetad set display as: `export DISPLAY=:0.1`, and so on.

Note: VirtualGL may be installed on servers with sudo access rights. It is not available on the tfpool.


### And
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


To start, run the Danijar Dreamer v1. It is based on tensorflow and MUJOCO.
It has 2KLOC

Note: MUJOCO requires a computer-tied and .edu email-tied license.

### MUJOCO
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

### dm_control
`pip install dm_control`  

