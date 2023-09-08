# Albert On isaacgym

## Environment dependencies
the installation is already made on yufu

If not, install isaacgym : https://developer.nvidia.com/isaac-gym

## How to launch the simulation : 

Launch the file : **Isaac/RLTests/testTensor.py**

## Structure: 
In its structure the project is much the same as for mujoco and pybullet, however the environment file is :

**Isaac/ClassFrameWorkTensor.py**

### The most part of used functions are the same besides for Raycasting and Collision Detection : 
raycasting casts 21 times ray_collision() to cast rays that detect objects by shooting virtual boxes and doing collision detection with an **AABB (axis-aligned bounding boxes)** algorithm

**AlbertCube.get_contact_points()**  uses an AABB algorithm too, therefore isn't suited for this program ( since albert can have a different orientation than the room )

## Details on ClassFrameWorkTensor



## Where we are on this project : 
### The program works, however : 
- since no raycasting function was made for isaacgym I had to handwrite it, therefore if **ray_collision()** is used,
the visual interface doesn't work ( this is why you'll find this part of the program in the form of comments in **raycasting()**)*
- In **RLTests/testTensor.py** : the function action_debug() used to manually move Albert around doesn't work (don't use the keyboard library, there's a way to work with key inputs in isaacgym)
 -> Therfore a tensor of zeros is fed to **pre_physics_step()** which makes Albert motionless
- Whatever step(dt) is set in **ClassFrameWork/AlbertEnv.set_sim_params()**, the issue of bad collisions ( alberts enters quite a lot other objects) persists
- The setup was made in hurry so only the first Room is available and is made manually in **AlbertEnv.build_basic_room()**
- A Replica is made in **ObjectsEnvironment/RoomTensor/Roop.py** with **build_basic_room()**

