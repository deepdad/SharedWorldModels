from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
import time

pr = PyRep()
# Launch the application with a scene file in headless mode
# pr.launch('PyRep/tests/assets/test_scene.ttt', headless=False)  # has an error
pr.launch('PyRep/examples/scene_panda_reach_target.ttt') #scene_with_panda.ttt')
pr.start()  # Start the simulation

# normally at the top
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
# change the scene
r = g = b = 1
w = h = d = 100
x = y = z = 50

print("no sure this works, I needed to move the GUI bar above the logged output at the bottom to see the 3D scene")
object = Shape.create(type=PrimitiveShape.CYLINDER,
                      color=[r,g,b], size=[w, h, d],
                      position=[x, y, z])
object.set_color([r, g, b])
object.set_position([x, y, z])

time.sleep(5)

print("no sure this works")

# use the Panda robot arm
arm = Panda()  # Get the panda from the scene
gripper = PandaGripper()  # Get the panda gripper from the scene

velocities = [.1, .2, .3, .4, .5, .6, .7]
arm.set_joint_target_velocities(velocities)
pr.step()  # Step physics simulation

done = False
# Open the gripper halfway at a velocity of 0.04.
while not done:
    done = gripper.actuate(0.5, velocity=0.04)
    pr.step()

pr.stop()  # Stop the simulation
print("wait a second to copy the log")
time.sleep(10)
pr.shutdown()  # Close the application
