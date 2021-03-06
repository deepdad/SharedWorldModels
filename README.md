# SharedWorldModels

Dream to Control: Learning Behaviors by Latent Imagination  
Mastering Atari with Discrete World Models

We are targeting RLBench with DreamerV1.
This idea was based on the assumption that DreamerV2 would not be suitable
 for continuous action spaces but when that turned out to be a mistake,
 the decision was rebased to using V1 being more useful for comparisons.
There remain parallelization and serial performance problems.

RLBench is a benchmark consisting of robot arm tasks.
For example reaching for a small black ball on a table (the default task in our main.py).  

Our preliminary question is whether we can get a DreamerV1 agent to complete the RLBEnch tasks,
 as they have sparse rewards upon task completion.

We assume that it can because DreamerV1 was also reported to be able to complete
 sparse reward tasks from DeepMind Control Suite.


We build on dreamer-pytorch by @juliusfrost: https://github.com/juliusfrost/dreamer-pytorch  
If RLBench works, we want to use a shared world model across robots.  



So we verify the (DreamerV1-tensorflow-) original mujoco/dmcontrol tasks with dreamer-pytorch.  
We run multiple RLBench tasks with dreamer-pytorch.  
We share a world model across tasks.

## Installation
Note: RLBench has a strict openGL>3 driver dependency.

#### Install python(3.8/9) dependencies
```bash
pip install -r requirements.txt
```

#### Atari ROMs
Atari is used for platform checks/tests.  

In order to import ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the `.rar` file.  Once you've done that, run:

`python -m atari_py.import_roms <path to folder>`

This should print out the names of ROMs as it imports them.  The ROMs will be copied to your `atari_py` installation directory so that you can run:

```bash
python atari_py_test.py /home/$(whoami)/venv/lib/python3.9/site-packages/atari_py/atari_roms/pong.bin
```


### RLBENCH
`pip install rlbench`  
(just a matter of time I guess)  

RLBench uses CoppeliaSim (formerly known as V-Rep), hence PyRep is still used.
This is a 3D simulator with a pluggable physics engine, but MUJOCO is not supported.
The benefit of that is that a MUJOCO license is not needed. The downside is that MUJOCO
 is the better physics engine for robotics tasks.  
Note: Most runtime bugs that we experience come from, for example not being able to calculate a nonlinearpath with PyRep/CSim.

For now, note that the folder (path)s:  
CoppeliaSim_Edu_V4_1_0_Ubuntu*/  
CoppeliaSim_Edu_V4_1_0_Ubuntu*.*.*  
RLBench/  
PyRep/  
are in .gitignore, you need to download these and put the expanded folders in the SharedWorldModels 
folder and in your .bashrc, before you can download and intsall PyRep, before you can download and 
install RLBench.  

Please see the README in here:  
git clone https://github.com/stepjam/PyRep

PyRep currently officially supports version **4.1** (we use 4.2's ReachTarget without problems,
but other tasks raise errors from CSim, there is a workaround in the PyRep changes folder 
but that only works for 4_1) of CoppeliaSim. This requires an 
OpenGL >3:
```bash
glxinfo | grep "OpenGL version"
```
This in turn requires a DISPLAY (see below).  

#### CoppeliaSim, PyRep

Download: 
- [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this. Also, do the same for other relevant users.

You can then pull PyRep from github:

```bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```
Install the PyRep python library:

```bash
pip install -r requirements.txt
pip install -e .
```

Try running one of the examples in the *examples/* folder.

_Although you can use CoppeliaSim on any platform, communication via PyRep is currently only supported on Linux._

#### RLPyt
RLPyt provides dreamer-pytorch with the samplers and runners for the RLBench Environments and tasks, for them to be run in CoppeliaSim.
Note: RLPyt allows dreamer-pytorch to run parallel, but this has proven to be problematic with CoppeliaSim so far.  

git clone https://github.com/astooke/rlpyt.git  
cd rlpyt  
pip install -r requirements  
pip install -e .  

### RLBench itself
Now head back to the RLBench folder, which is located in this repo, but that being a git repo itself,
its changes are not tracked by the SharedWorldModels repo.  
The relative path in the following assumes that RLBench is at SharedWorldModels/RLBench.  

To finish or to update the RLBench [installation](https://github.com/stepjam/RLBench#install),
in SharedWorldModels/RLBench:
```bash
pip install -r requirements.txt #only when needed
pip install -e .
```

#### Making changes to rlbench, rlpyt or other third party repos
When making changes in RLBench, they need to be copied to where python finds them. This may be the venv python.
So, what I do is, for example, I make two copies:
```bash
cp RLBench/rlbench/tasks/reach_target.py rlbench_changes/original/
cp RLBench/rlbench/tasks/reach_target rlbench_changes/
```
make changes to reach_target.py and save those in rlbench_changes:
```
cp rlbench_changes/original/readch_target.py RLBench/rlbench/tasks/reach_target.py 
```
**when working with PyCharm, there may be a different practice.

I could just save them in RLBench directly but I think it may be better to use symlinks there in the future.
so that the reach_target.py file is read via a symlink from the SharedWorldModels/rlbench_changes
 folder. This allows making changes in the RLBench repo that are tracked by the SWM repo, similar to
how Ray does this.

* We have been using git branches for experiments, but intend to migrate to store and document them in the experiments
  folder.

### DISPLAY
There are several options to get a DISPLAY (or forego one):  
1. Run locally (need a CUDA device, preferably with 16GB GPU memory).  
2. Use a remote desktop with VNC (not on tfpool).  
3. ssh -X (-C for compression? /=slow)  
4. Run headless:  
4.1 Use xvfb-run python main.py (slow) (on tfpool via poolmgr (Sascha Frank may not be helpful)).  
4.2 Use VirtualGL (not seen to work yet).  

##### Running Headless

If you plan to run on a headless machine, to run with a virtual framebuffer, e.g.:

```bash
sudo apt-get install xvfb
xvfb-run python3 my_pyrep_app.py
# or if you are using jupyter
# xvfb-run jupyter notebook
```
You can run RLBench headlessly with VirtualGL. VirtualGL is an open source toolkit that gives any Unix or Linux remote display software the ability to run OpenGL applications **with full 3D hardware acceleration**.
This works, but you need to have a good GPU yourself, or OpenGL at least locally it seems.
It's possible to set the steps per frame for CSim (8 by default); a slow display of the simulation seems to hold up the simulation.
First insure that you have the nVidia proprietary driver installed. I.e. you should get an output when running `nvidia-smi`. Now run the following commands:
```bash
sudo apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

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

I have preferred to use VNC but that is costlier than preparing a set of experiments locally and only spinning up a GPU VM to run or scale them.
### Tooling
#### nvtop
```bash
sudo apt install cmake libncurses5-dev libncursesw5-dev  
git clone https://github.com/Syllo/nvtop.git  
mkdir -p nvtop/build && cd nvtop/build  
cmake ..  
make  
```
Install globally on system  
```bash
sudo make install  
```
#### PyCharm
unpack and run  
use the venv configuration in the Run configuration
#### Remote Desktop
GCloud is set up with VNC. Details via e-mail.

#### Task Builder
run 
```bash
python RLBench/rlbench/tools/task_builder.py
```
then use the commands listed in the terminal. You can't use .ttt files saved by saving from the File menu in CSim. 
You must save .ttm files using the terminal. Renaming is best, but it will also move the original python file you
choose to start from. That's why we save a backup in the rlbench_changes/backup folder.
Doing this will allow you to import camel case tasks (`from rlbench.tasks import FastSingle2xTarget`)in the custom rlbench.py environment or in main.
* If you get errors about a missing handle, then TaskName.py will use variables that are not in theCSim model.
* If the error is that TaskName task is missing in `__init__.py` then you can add it there manually.
* At the top you can set the ms steps for the physics engine (and its speed setting, but this seems sticky/buggy)

## R&D
To run with RLBench, run `python main.py --help`. add arguments (HP's in code).  

You can use tensorboard.
Run `tensorboard --logdir=data`.

### Testing
To run tests:
```bash
pytest tests
```

### Getting started
To start, run the Danijar Dreamer v1. It is based on tensorflow and MUJOCO.

Note: MUJOCO requires a computer-tied and .edu email-tied license.

### MUJOCO
The instructions below for MUJOCO, patchelf and mesa can be read from the trace of trying:
`pip install mujoco_py`  

in ~/.bashrc:  

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$(whoami)/.mujoco/mujoco200/bin

```bash
cp mujoco200_linux/ .mujoco/ -r  
cd .mujoco/  
mv mujoco200_linux/ mujoco200  
place your license key (the mjkey.txt file from your email) at ~/.mujoco/mjkey.txt  

sudo apt install patchelf  
```
or from [source](https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz):

```bash
make  
./configure  
make  
sudo make install  
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

For the time being, tf2.5 requiring numpy < 1.20:  
`pip install -U numpy`

Now it should work:  
`pip install mujoco_py`  

`$python3`
`>>> import mujoco_py`   
`>>> import gym`  
`>>> env = gym.make('FetchReach-v1')`  
`>>> env.render()`

### dm_control
`pip install dm_control`
