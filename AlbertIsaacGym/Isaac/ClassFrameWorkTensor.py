import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
from Isaac.ObjectsEnvironmentTensor.RoomManagerTensor import RoomManager
from Isaac.ObjectsEnvironmentTensor.AlbertTensor import AlbertCube
from Isaac.ObjectsEnvironmentTensor.RoomTensor import Room
import math
import numpy as np
class AlbertEnvironment():

    # Common callbacks

    def create_sim(self):
        # create environments and actors
        self.gym = gymapi.acquire_gym()



        args = gymutil.parse_arguments()
        # configure sim params
        sim_params = self.set_sim_params()

        # create sim with these parameters
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # configure the ground plane
        self.configure_ground()

        #####################  ASSETS  #####################

        asset_albert, asset_room = self.prepare_assets()

        asset_options_base_cube = gymapi.AssetOptions()
        asset_options_base_cube.fix_base_link=True
        asset_base_cube = self.gym.create_box(self.sim, width=0.5, height=0.5, depth=0.5,
                                         options=asset_options_base_cube)

        asset_options_door = gymapi.AssetOptions()
        asset_options_door.fix_base_link=True
        asset_door = self.gym.create_box(self.sim, width=0.5, height=0.5, depth=0.5, options=asset_options_door)

        asset_options_button = gymapi.AssetOptions()
        asset_options_button.fix_base_link=True
        asset_button = self.gym.create_box(self.sim, width=0.5, height=0.5, depth=0.01, options=asset_options_button)

        #####################  ENVIRONMENT SETTING  #####################

        # set up the env grid
        self.num_envs = 2
        envs_per_row = 5
        env_spacing = 20.0
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # cache some common handles for later use
        envs = []
        self.actor_handles = []
        self.albert_handle=[]
        # instanciate Room Manager
        self.room_manager = RoomManager()
        self.albert_array = np.empty((self.num_envs,))

        # create and populate the environments
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            envs.append(env)

            pose_albert = gymapi.Transform()
            pose_albert.p = gymapi.Vec3(2.0, 3.0, 0.501)  # pose.r pour l'orientation
            # la position est relative
            actor_handle_albert = self.gym.create_actor(env, asset_albert, pose_albert, "Albert", i)  # creation  d'un acteur à partir d'un asset
            self.albert_handle.append(actor_handle_albert)
            self.actor_handles.append(actor_handle_albert)

            #pose_room = gymapi.Transform()
            #pose_room.p = gymapi.Vec3(0.0, 0.0, 0.0)  # pose.r pour l'orientation
            # la position est relative
            #actor_handle_room = self.gym.create_actor(env, asset_room, pose_room, "Room", i,
             #                                         1)  # creation  d'un acteur à partir d'un asset
            #self.actor_handles.append(actor_handle_room)

            self.num_bodies = self.build_basic_room(env,asset_base_cube,asset_door,i)
            pose_button=gymapi.Transform()
            pose_button.p=gymapi.Vec3(1.5,2,0.5)
            actor_handle_button=self.gym.create_actor(env,asset_button,pose_button,"button",i,1)
            self.actor_handles.append(actor_handle_button)

        # prepare simulation buffers and tensor storage - required to use tensor API
        self.gym.prepare_sim(self.sim)

        # Viewer Creation
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)

        #####################  ACQUIRE TENSORS  #####################
        # forces de contact
        _net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_contact_force_tensor = gymtorch.wrap_tensor(_net_contact_force_tensor)

        # pos,ori,vel,ang_vel des root ( a voir comment on fait si la compo de la room n'est pas full root
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)

        ############################### Creation of the Room and Albert objects #########################################
        print(" ENVS : " + str(envs))
        print("albert handle : "+str(self.albert_handle))
        self.room_manager.add_room(Room(self.num_bodies,self.num_envs))
        self.albert_tensor = AlbertCube(self.sim,self.gym,self.viewer,self.room_manager,self.num_bodies,envs,self.root_tensor,torch.tensor(self.albert_handle))
        

        self.time_passed = torch.zeros((self.num_envs,))
        self.time_episode = 10
        self.step = 0.01  # dt
        self.curr_state = self.get_current_state()
        self.prev_state = self.get_previous_state()
        self.actions = None

    def build_basic_room(self, env, asset_base_cube, asset_door,collision_group):  # construction de la structure de la chambre et stockage des blocs dans une liste
        x, y, l = 0, 0, 0.25
        depth = 6
        width = 11
        height = 3
        id=1 ############# PEUT ETRE UN PB DE COMMENCER A ID = 1
        for i in range(depth):
            for j in range(width):
                pose_floor = gymapi.Transform()
                pose_floor.p = gymapi.Vec3(x + i/2, y + j/2, l)  # pose.r pour l'orientation
                name = "cube" + str(id)
                id+=1
                actor_handle_cube = self.gym.create_actor(env, asset_base_cube, pose_floor,name, collision_group,1)
                self.actor_handles.append(actor_handle_cube)

                for z in range(height):  # MURS
                    if i == 0 or (j == 0 or j == 10):
                        if i == depth / 2 and (j == width - 1 or j == 0) and (z == 0):
                            if j == width - 1:
                                pose_door = gymapi.Transform()
                                pose_door.p = gymapi.Vec3(x + i/2, y + j/2, l + 1/2 + z/2)  # pose.r pour l'orientation
                                name = "door"
                                id+=1
                                actor_handle_door = self.gym.create_actor(env, asset_door, pose_door, name, collision_group,1)
                                self.actor_handles.append(actor_handle_door)
                        else:
                            pose_wall = gymapi.Transform()
                            pose_wall.p = gymapi.Vec3(x + i/2, y + j/2, l + 1/2 + z/2)  # pose.r pour l'orientation
                            name = "cube" + str(id)
                            id+=1
                            actor_handle_wall = self.gym.create_actor(env, asset_base_cube, pose_wall, name, collision_group,1)
                            self.actor_handles.append(actor_handle_wall)
        return id

    def pre_physics_step(self, actions):######################## FINI ##############################
        # apply actions
        self.actions=actions # JE PEUX FAIRE CA ??? ############################ ISAAC ######################################
        self.albert_tensor.take_action(actions)


    def post_physics_step(self):################# FINI ######################
        # compute observations,rewards,and resets
        self.compute_observations()
        self.compute_reward()

        reset_tensor = ( self.time_passed >= self.time_episode ) | ( self.albert_tensor.has_fallen() ) | ( self.achieved_maze() )

        self.time_passed += self.step

        self.reset(reset_tensor)

    def reset(self, reset_tensor):
        room_tensor=self.room_manager.room_array[self.albert_tensor.actual_room]
        room_tensor.reset_room(self.root_tensor,self.albert_tensor,reset_tensor)

        pos = torch.tensor([1+2*torch.rand(1),1+4*torch.rand(1),0.75]).repeat(self.num_envs,1)
        ori = torch.tensor([0,0,torch.rand(1)*2*torch.pi - torch.pi]).repeat(self.num_envs,1)

        ori = euler_to_quaternion(ori)

        self.root_tensor[self.albert_tensor.id_array][:,:3]=torch.where(reset_tensor.unsqueeze(1).repeat(1,3).type(torch.bool),pos,self.root_tensor[self.albert_tensor.id_array][:,:3])
        self.root_tensor[self.albert_tensor.id_array][:,3:7] = torch.where(reset_tensor.unsqueeze(1).repeat(1,4).type(torch.bool), ori,self.root_tensor[self.albert_tensor.id_array][:,3:7])

        self.albert_tensor.reset_memory_state(reset_tensor)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.compute_observations()
        self.time_passed[reset_tensor]=0


    def compute_observations(self):##################### FINI ##############################
        # refresh state tensor
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.obs_buf = self.albert_tensor.get_observation()

        self.update_state()


    def compute_reward(self):##################### FINI #########################
        reward = torch.zeros((self.num_envs,))
        contact = self.curr_state["contactPoints"] # regarder comment modifier la space du State courant
        reward = torch.where(self.actions[:,2]==1,reward-0.05,reward) # si il saute
        reward = torch.where(torch.any(((contact==3) | (contact==4) | (contact==5) ),axis=1), reward - 0.1, reward)# FINIR######################S  # si contact avec des obstacles
        reward = torch.where(self.achieved_maze(), reward + 1, reward)  #
        reward = torch.where(self.button_distance() != 0, reward + 1, reward)  #MODIFIER LA FONCITON BUTTONDISTANCE
        reward = torch.where(self.albert_tensor.has_fallen(), reward - 0.05, reward)  # si il saute

        return reward


    def button_distance(self):# tout est à modifier ############################## FINI #########################
        n = len(self.curr_state["buttonsState"])
        modified_prev_state = torch.where(self.prev_state["buttonsState"]==math.nan,0,self.prev_state["buttonsState"])
        d_sum = torch.sum(torch.abs(self.curr_state["buttonsState"]-modified_prev_state),axis=0)
        return d_sum


    def achieved_maze(self):######################## FINI ##########################
        door_id_tensor = self.room_manager.room_array[self.albert_tensor.actual_room].door_array_tensor.id_tensor

        character_pos = self.albert_tensor.get_pos_tensor()[:,:2]# on veut que les pos x et y donc pas besoin de prendre z
        door_pos = self.root_tensor[door_id_tensor][:,:2]
        dist = torch.sum((character_pos-door_pos)**2,axis=1)

        return (dist < 0.5)  # pour l'instant 0.5 mais en vrai dépend de la dim de la sortie et du character

    def prepare_assets(self):############## FINI CAR RIEN A CHANGER ########################
        project_name = "testIsaac"
        project_path = get_absolute_path_project(project_name).replace('\\', '/')
        xml_directory_path = project_path + "/assets/"

        # Asset 1 : Albert :

        asset_file_albert = "Albert.xml"  # a changer avec le fichier MJCF
        asset_options_albert = gymapi.AssetOptions()
        asset_options_albert.fix_base_link = False  # a voir ce que c'est
        asset_options_albert.armature = 0.01  # a voir aussi


        asset_albert = self.gym.load_asset(self.sim, xml_directory_path, asset_file_albert, asset_options_albert)

        # Asset 2 : Room :

        asset_file_room = "Room.xml"  # a changer avec le fichier MJCF
        asset_options_room = gymapi.AssetOptions()
        asset_options_room.fix_base_link = True  # a voir ce que c'est
        asset_options_room.armature = 0.01  # a voir aussi

        asset_room = self.gym.load_asset(self.sim, xml_directory_path, asset_file_room, asset_options_room)

        return asset_albert, asset_room

    def configure_ground(self):############## FINI CAR RIEN A CHANGER ########################
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0

        # create the ground plane
        self.gym.add_ground(self.sim, plane_params)

    def set_sim_params(self):############## FINI CAR RIEN A CHANGER ########################
        # get default set of parameters
        sim_params = gymapi.SimParams()

        # set common parameters
        sim_params.dt = 0.01
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        # set PhysX-specific parameters
        sim_params.physx.use_gpu = True
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0

        # set Flex-specific parameters
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20
        sim_params.flex.relaxation = 0.8
        sim_params.flex.warm_start = 0.5

        return sim_params

    def get_current_state(self):#################### FINI ############################
        current_state = self.albert_tensor.current_state
        return current_state

    def get_previous_state(self):###################### FINI ########################
        prev_state = self.albert_tensor.get_previous_state()
        return prev_state

    def update_state(self):############################ FINI ############################
        self.curr_state = self.get_current_state()
        self.prev_state = self.get_previous_state()



def get_absolute_path_project(project_name):

    script_directory = os.path.dirname(os.path.abspath(__file__))

    current_directory = script_directory
    while current_directory != os.path.dirname(current_directory):
        if os.path.basename(current_directory) == project_name:
            return current_directory
        current_directory = os.path.dirname(current_directory)

    # If the specified directory is not found, return None
    return None


def euler_to_quaternion(euler_angles):##################### FINI ######################
    """
    Convert a 1D tensor of XYZ Euler angles to a tensor of quaternions.

    Args:
        euler_angles (torch.Tensor): 1D tensor of Euler angles in radians with shape (3,).

    Returns:
        torch.Tensor: Tensor of quaternions with shape (4,).
    """
    roll, pitch, yaw = euler_angles[:,0], euler_angles[:,1], euler_angles[:,2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    quaternion = torch.stack((w, x, y, z),dim=1)
    return quaternion