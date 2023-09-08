# Albert On isaacgym


On the computer Yufu, this project is located at :

    ~/git/smoyal/isaacgymEnvi/isaacgymEnvi/testIsaac

## Environment dependencies
The installation is already made on yufu:

If not, install isaacgym : https://developer.nvidia.com/isaac-gym

Download the zip forlder, go to isaacgym/docs/index.html for all of the documentation
## How to launch the simulation : 

Launch the file : **Isaac/RLTests/testTensor.py**

## Structure: 
In its structure the project is much the same as for mujoco and pybullet, however the environment file is :

**Isaac/ClassFrameWorkTensor.py**

### The most part of used functions are the same besides for Raycasting and Collision Detection : 
raycasting casts 21 times ray_collision() to cast rays that detect objects by shooting virtual boxes and doing collision detection with an **AABB (axis-aligned bounding boxes)** algorithm

**AlbertCube.get_contact_points()**  uses an AABB algorithm too, therefore isn't suited for this program ( since albert can have a different orientation than the room )

## A key difference : 
### we work now with multiple rooms simultaneously : 
Therefore, even though there is still a unique object Albert and a unique Object Room,
every one of their former scalar component is now a torch tensor of size (number_of_environments)

### To reset environments, we use a tensor:
This tensor reset_tensor is a boolean tensor of size (number_of_environments)
 with **reset_tensor[i] == True ** if the environment i needs to be reset

## Details on ClassFrameWorkTensor.py:

### self.create_sim():
#### Prepares The gym, the simulation and all of the environments:

1 - Gym and sim creation

2 - plane configuration

3 - Assets creation ( blueprints for actors )

4 - Environment Settings : 
- self.num_envs = number of environments to simulate
- handles are arrays of unique id for each object
- in the for loop, all the environments are created :
  - the environment is created
  - then Albert is created
  - self.build_basic_room() is called to create the whole room

5 - Acquiring the needed tensors:
  - root_tensor is the tensor containing all the information (position,orientation,linear and angular velocity) of each root actor

6 - Creation of the Room and Albert objects

### Environment functions : 
the function **step()** is on isaac gym separated into 2 components : 

1 - pre_physics_step(action) : computes the action

2 - post_pysics_step() : computes the observations, the reward and resets

the function **compute_observations()** : 

1 - Refreshes the root tensor to get the values post action

2 - computes Albert's Observations




## Where we are on this project : 
### The program works, however : 
- since no raycasting function was made for isaacgym I had to handwrite it, therefore if **ray_collision()** is used,
the visual interface doesn't work ( this is why you'll find this part of the program in the form of comments in **raycasting()**)*
- In **RLTests/testTensor.py** : the function action_debug() used to manually move Albert around doesn't work (don't use the keyboard library, there's a way to work with key inputs in isaacgym)
 -> Therefore a tensor of zeros is fed to **pre_physics_step()** which makes Albert motionless
- Whatever step(dt) is set in **ClassFrameWork/AlbertEnv.set_sim_params()**, the issue of bad collisions ( alberts enters quite a lot in  other objects) persists
- The setup was made in hurry so only the first Room is available and is made manually in **AlbertEnv.build_basic_room()**
- A Replica is made in **ObjectsEnvironment/RoomTensor/Room.py** with **build_basic_room()**

