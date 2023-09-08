
from isaacgym import gymapi,gymtorch
import torch
import torch.nn.functional as F
import math
import numpy as np
from Isaac.ObjetsEnvironnementTensor.CubeTensor import Cube

# Classe de l'Acteur : Albert
class AlbertCube(Cube):

    def __init__(self, sim,gym,viewer, room_manager, num_bodies, env, state_tensor, handle_albert_tensor):
        # super().__init__(hExtents=[0.25,0.25,0.25])
        self.actual_room = 0  # niveau actuel d'entrainement dans la liste du room manager
        self.room_manager = room_manager  # classe contenant la liste de tous les niveaux d'entraînement possibles
        self.num_bodies = num_bodies
        self.id_array = torch.tensor([i * num_bodies for i in range(len(env))])
        self.time = torch.zeros((len(env),))  # temps passé dans la simu depuis sa création

        # Caracs of simulation
        self.state_tensor = state_tensor
        self.gym = gym
        self.sim = sim
        self.viewer = viewer
        self.env = env
        self.num_envs = len(env)
        self.handle_albert_tensor = handle_albert_tensor
        # espace d'état ( albert n'y a pas "acces")
        self.memory_state = self.init_memory_state()  # stockage des 5 derniers états
        self.current_state = self.get_current_state()  # état courant de la simulation

        # espace d'observation (albert y a accès )
        self.memory_observation = torch.full((5,self.num_envs,42),-1)  # stockage des 5 dernieres observations

        # Attributs nécessaires aux mouvements d'albert

        self.x_factor = torch.zeros((self.num_envs,))  # pour le saut
        self.jumping = torch.full((self.num_envs,), False)  # pour le saut

    def has_fallen(self):  # retourne True si Albert est tombé du niveau ########################### FINI #########################
        pos = self.get_pos_tensor()
        room_tensor = self.room_manager.room_array[self.actual_room]  ############# ligne a changer evidement apres

        has_fallen_tensor = pos[:,2]<room_tensor.global_coord[:,2]

        return has_fallen_tensor

    def reset_time(self,reset_tensor):  ################################## FINI #########################################
        self.time = torch.where(reset_tensor,0,self.time)

    def reset_pos_ori(self, pos,ori_euler):  #################################### FINI ###############################
        ori_quaternion = euler_to_quaternion(ori_euler)
        self.state_tensor[self.id_array][:3] = pos
        self.state_tensor[self.id_array][3:7] = ori_quaternion


    def raycasting(self):  # Not the best implementation of rays, makes the visual simulation impossible
        cube_pos = self.get_pos_tensor()
        cube_ori = self.get_ori_tensor()
        ray_vects = self.grid_vision(cube_pos, cube_ori, ray_length=10)  # définit le quadrillage par les rayons

        # RAYCASTING
        """
        contact_results = torch.tensor([])
        for n in range(21):
            print("raycasting 6_"+str(n))
            contact_results=torch.cat((contact_results,
            self.ray_collision(point_of_origin_tensor=cube_pos,end_pos_ray=ray_vects[n]).unsqueeze(0)))  # fonction de raycasting
            
        for n in range(21):
            print("raycasting 8_"+str(n))
            contact_results[n,:,0] = self.check_type(contact_results[n,:,0].unsqueeze(-1)).squeeze(-1)

        """

        desired_value_dummy = torch.tensor([0,10])
        dummy_shape = (21,self.num_envs,2)
        contact_results = desired_value_dummy.view(1,1,2).expand(dummy_shape).to(torch.float)
        contact_results
        #self.show_grid(cube_pos,ray_vects) ## UNCOMMENT THIS LINE FOR VISIBLE RAYCASTING
        return contact_results


    def grid_vision(self,character_pos, character_ori,ray_length):  # retourne la position du bout des tous les rayons nécessaires à la vision #################### FINI #######################
        cube_ori = quaternion_to_euler(character_ori)
        matrice_ori = euler_to_rotation_matrix(cube_ori)

        # On détermine ici les angles des rayons pour le quadrillage
        # départ des angles :
        dep_angles_yaw = -35 * torch.pi / 180
        dep_angles_pitch = -10 * torch.pi / 180
        # Pas yaw pour 70°
        step_yaw = 70 / 6
        step_yaw_rad = step_yaw * torch.pi / 180

        # pas pitch pour 70°
        step_pitch = 20 / 2
        step_pitch_rad = step_pitch * torch.pi / 180

        # rayVec1 : premier rayon droit devant le cube
        ray_vects = []
        for i in range(3):
            for n in range(7):
                base_ray = torch.tensor([np.cos((n * step_yaw_rad + dep_angles_yaw)) * np.cos((i * step_pitch_rad + dep_angles_pitch)),
                            np.sin((n * step_yaw_rad + dep_angles_yaw)), np.sin((i * step_pitch_rad + dep_angles_pitch))])
                norm_ray = torch.linalg.norm(base_ray)
                
                


                ray_tensor=torch.tensor(
                    [(base_ray[0] / norm_ray * ray_length),
                    (ray_length * base_ray[1] / norm_ray),
                    (ray_length * base_ray[2] / norm_ray)
                    ]
                ).expand(self.num_envs,-1)
                

                matrice_ori=matrice_ori.view(-1,3,3).to(torch.float)
                ray_tensor=ray_tensor.view(-1,3,1).to(torch.float)
                
                a=torch.matmul(matrice_ori,ray_tensor)
                a=a.view(-1,3)


                a[:,0] += character_pos[:,0]
                a[:,1] += character_pos[:,1]
                a[:,2] += character_pos[:,2]

                ray_vects.append(a)
        ray_vects=torch.stack(ray_vects)
        
        return ray_vects



    def ray_collision(self, point_of_origin_tensor, end_pos_ray):################################## FINI #####################################
        room_tensor = self.room_manager.room_array[self.actual_room]
        distance = end_pos_ray - point_of_origin_tensor
        id_distance_collision_tensor = torch.tensor([-1,10])
        id_distance_collision_tensor = id_distance_collision_tensor.repeat(self.num_envs,1)
        to_check_tensor = torch.full((self.num_envs,), False)
        for i in range(100):
            ray_pos = point_of_origin_tensor + distance * i / 100
            Amin_ray = ray_pos - 0.1
            Amax_ray = ray_pos + 0.1

            for button in room_tensor.buttons_array_tensor:
                button_pos = self.state_tensor[button.id_tensor][:,:3]
                Amin_button = button_pos - torch.tensor([0.5, 0.5, 0.1])
                Amax_button = button_pos + torch.tensor([0.5, 0.5, 0.1])
                result = check_collision_AABB(to_check_tensor, Amin_button, Amax_button, Amin_ray, Amax_ray)
                
                id_distance_collision_tensor = torch.where(result.unsqueeze(1).expand(-1,2),
                    torch.cat((button.id_tensor.view(self.num_envs,1), torch.tensor([10 * i / 100]).repeat(self.num_envs,1)), axis=1), id_distance_collision_tensor)
                to_check_tensor = id_distance_collision_tensor[:, 0] == -1

            for box_id in room_tensor.floor_array_tensor:
                box_pos = self.state_tensor[box_id.to(torch.int)][:,:3]
                Amin_box = box_pos - 0.5
                Amax_box = box_pos + 0.5
                result = check_collision_AABB(to_check_tensor, Amin_box, Amax_box, Amin_ray, Amax_ray)
                id_distance_collision_tensor =torch.where(result.unsqueeze(1).expand(-1,2),
                    torch.cat((box_id.view(self.num_envs,1), torch.tensor([10 * i / 100]).repeat(self.num_envs,1)), axis=1), id_distance_collision_tensor)
                to_check_tensor = id_distance_collision_tensor[:, 0] == -1

            for box_id in room_tensor.wall_array_tensor:
                box_pos = self.state_tensor[box_id.to(torch.int)][:,:3]
                Amin_box = box_pos - 0.5
                Amax_box = box_pos + 0.5
                result = check_collision_AABB(to_check_tensor, Amin_box, Amax_box, Amin_ray, Amax_ray)
                id_distance_collision_tensor =torch.where(result.unsqueeze(1).expand(-1,2),
                    torch.cat((box_id.view(self.num_envs,1), torch.tensor([10 * i / 100]).repeat(self.num_envs,1)), axis=1), id_distance_collision_tensor)
                to_check_tensor = id_distance_collision_tensor[:, 0] == -1

            # for door :
            door_pos = self.state_tensor[room_tensor.door_array_tensor.id_tensor][:,:3]
            Amin_box = door_pos - 0.5
            Amax_box = door_pos + 0.5
            result = check_collision_AABB(to_check_tensor, Amin_box, Amax_box, Amin_ray, Amax_ray)
            id_distance_collision_tensor =torch.where(result.unsqueeze(1).expand(-1,2),
                    torch.cat((room_tensor.door_array_tensor.id_tensor.view(self.num_envs,1), torch.tensor([10 * i / 100]).repeat(self.num_envs,1)), axis=1), id_distance_collision_tensor)
            to_check_tensor = id_distance_collision_tensor[:, 0] == -1

        return id_distance_collision_tensor

    def show_grid(self,cube_pos,ray_vects): ################################### FINI #####################################################

        for i in range(self.num_envs):

            all_pos = torch.array([])
            for n in range(21):
                all_pos = torch.cat(all_pos, torch.cat(cube_pos[i], ray_vects[n][i]))

            rgb = torch.array([1, 0, 0])

            self.gym.add_lines(self.viewer, self.env[i], 21, all_pos, rgb)


    def jump_zer(self, jump, move):  ############################ FINI ##################################
        i = 13000  # force du jump sur un pas
        minus_ones_tensor = torch.full((self.num_envs,), -1)
        ones_tensor = torch.full((self.num_envs,), 1)
        zero_tensor = torch.zeros((self.num_envs,))
        move_x = torch.where(move == zero_tensor, zero_tensor,
                             torch.where(ones_tensor == move, minus_ones_tensor, ones_tensor))

        self.jumping = torch.where(jump == ones_tensor & self.in_contact_with_floor_or_button(),
                                   torch.full((self.num_envs,), True), self.jumping)

        self.x_factor = move_x
        ori_tensor = self.get_ori_tensor()
        self.oriJump = quaternion_to_euler(
            ori_tensor)  ############## pour l'instant osef mais enft on ca peut etre l'utiliser pour direct changer de referentiel dans la force
        stack1 = torch.cat((torch.zeros(self.num_envs,1),
                              (jump * i * torch.where(self.in_contact_with_floor_or_button(),
                                                    torch.ones((self.num_envs)), torch.zeros((self.num_envs)))).unsqueeze(1)),
                             dim=1)  ######## changer in_contact_with_floor
        
        impulse = torch.cat(((move_x * 500).unsqueeze(1), stack1), dim=1)
        for i in range(self.num_envs):
            force = gymapi.Vec3(impulse[i][0],impulse[i][1],impulse[i][2])
            self.gym.apply_body_forces(env=self.env[i], rigidHandle=self.handle_albert_tensor[i].item(), force=force,
                                       torque=None, space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def yaw_turn(self,
                 rotate):  # fonction de rotation d'albert ############### pas le choix, je suis passé par un "in range"############# FINI #########################

        move_z = torch.where(rotate == 0, 0,torch.where(rotate == 1, -1, 1))
        angular_force = torch.cat((torch.zeros(( self.num_envs,2)), (move_z * 10).unsqueeze(1)), dim=1)
        for i in range(self.num_envs):
            torque = gymapi.Vec3(angular_force[i][0],angular_force[i][1],angular_force[i][2])
            self.gym.apply_body_forces(env=self.env[i], rigidHandle=self.handle_albert_tensor[i].item(), force=None,
                                       torque=torque, space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def move(self, move):  ############################## FINI ###########################
        minus_ones_tensor = torch.full((self.num_envs,), -1)
        ones_tensor = torch.full((self.num_envs,), 1)
        zero_tensor = torch.zeros((self.num_envs,))
        move_x = torch.where(move == 0, 0,
                             torch.where(move == 1, -1, 1))

        linear_velocity = torch.cat(((move_x * 500).unsqueeze(1), torch.zeros((self.num_envs,2))), dim=1).expand(self.num_envs,-1)
        ori_tensor = self.get_ori_tensor()
        euler = quaternion_to_euler(ori_tensor)
        matrix_ori = euler_to_rotation_matrix(euler)
        matrix_ori=matrix_ori.view(-1,3,3).to(torch.float)
        linear_velocity = linear_velocity.view(-1,3,1).to(torch.float)
                
        linear_velocity = torch.matmul(matrix_ori, linear_velocity).view(-1,3)
        
    
        contact_floor_button = self.in_contact_with_floor_or_button()
        contact_binary = torch.where(contact_floor_button.unsqueeze(1).repeat(1,3).type(torch.bool) , torch.full((self.num_envs, 3), 1),
                                    torch.zeros((self.num_envs, 3)))
        linear_velocity = linear_velocity * contact_binary  # si on est dans les airs, ca doit valoir 0

        # impulse = torch.cat((np.array(linear_velocity), torch.zeros((self.num_envs,3))),dim=0) # dépend de la size de force enft
        impulse = linear_velocity

        for i in range(self.num_envs):
            force = gymapi.Vec3(impulse[i][0],impulse[i][1],impulse[i][2])
            self.gym.apply_body_forces(env=self.env[i], rigidHandle=self.handle_albert_tensor[i].item(), force=force,
                                       torque=None, space=gymapi.CoordinateSpace.LOCAL_SPACE)


    def take_action(self, action):  # 1: rotate, 2 : move, 3 : jump #################### FINI #########################
        rotate = action[:, 0]
        move = action[:, 1]
        jump = action[:, 2]
        self.yaw_turn(rotate)
        self.move(move)
        self.jump_zer(jump, move)
        self.current_state = self.get_current_state()  ########## PAS SUR ? ##################

    def get_observation(self): ######################## FINI ########################
        contact_results = self.raycasting()
        # Si type 0 ou -1, type devient 0 et distance devient 10
        condition = (contact_results[:, :, 0] == 0) | (contact_results[:, :, 0] == -1)
        contact_results[condition,:]=torch.tensor([0,10]).to(torch.float)
        contact_results_reshaped = torch.reshape(contact_results,(self.num_envs,42)) #reshaped but hasn't the right organization
        even_indices = contact_results_reshaped[:,0::2]
        odd_indices = contact_results_reshaped[:,1::2]
        contact_results_reshaped = torch.cat((even_indices,odd_indices),dim=1)# now all types are together first than come the distances
        self.add_to_memory_observation(contact_results_reshaped)
        observation = self.flat_memory()
        return observation

    def check_type(self, id_tensor):  ################### FINI ############################
        room_tensor = self.room_manager.room_array[self.actual_room]
        type_array = []
        for i in range(self.num_envs):
            type_sub_array = []
            for j in range(id_tensor[i].numel()):
                type_sub_array.append(
                    self.check_type_(id_tensor[i][j].item(),
                    room_tensor,
                    i))
            type_array.append(type_sub_array)
            
        type_tensor = torch.tensor(type_array)
        return type_tensor

    def check_type_(self, id,
                    room_tensor,i):  # retourne à quel type d'objet l'id fait référence ####################### FINI ###################### PAS SUR ENFAIT
        buttons = [room_tensor.buttons_array_tensor[j].id_tensor[i] for j in range(len(room_tensor.buttons_array_tensor))]
        if id in buttons:
            return 1

        if id in room_tensor.floor_array_tensor[:,i]:
            return 2

        if id in room_tensor.wall_array[:,i]:
            return 3

        if id in room_tensor.fences_array_tensor[:,i]:######## CHANGER en id_tensor
            return 4

        if id in room_tensor.iblocks_array_tensor[:,i]:########## changer en id_tensor
            return 5
        if id == room_tensor.door_array_tensor.id_tensor[i]:
            return 6
        return 0

    def add_to_memory_observation(self,current_observation):  # ajout de l'observation courante à la liste des 5 dernieres observations ############## FINI #####################
        self.memory_observation[0] = self.memory_observation[1]
        self.memory_observation[1] = self.memory_observation[2]
        self.memory_observation[2] = self.memory_observation[3]
        self.memory_observation[3] = self.memory_observation[4]
        self.memory_observation[4] = current_observation

    def add_to_memory_state(self, current_state):  # ajout de l'état courant du système à la liste des 5 derniers états #################### FINI ########################
            self.memory_state[0] = self.memory_state[1]
            self.memory_state[1] = self.memory_state[2]
            self.memory_state[2] = self.memory_state[3]
            self.memory_state[3] = self.memory_state[4]
            self.memory_state[4] = current_state


    def get_previous_state(self):##################### FINI ############################
        return self.memory_state[-2]

    def get_current_state(self):  # fonction actualisant l'état courant du système et retournant les 5 derniers états ################## MOYEN FINI ########################
        room_tensor = self.room_manager.room_array[self.actual_room]
        current_state = {}
        pos_albert = self.get_pos_tensor()
        buttons_tensor = room_tensor.buttons_array_tensor ######################## A VOIR DANS LES MODIFS DE ROOM / A VOIR DANS CHECK TYPE

        buttons_tensor = binarize(buttons_tensor)
        door_tensor = torch.prod(buttons_tensor,dim=0)
        door_pos_tensor = self.state_tensor[room_tensor.door_array_tensor.id_tensor][:,:2]

        current_state["CharacterPosition"] = pos_albert
        current_state["doorState"] = door_tensor.to(torch.float)
        current_state["doorPosition"] = door_pos_tensor
        current_state["buttonsState"] = buttons_tensor.to(torch.float)

        # add contactpoints
        contact_points = self.get_contact_points() #  PB AVEC CETTE FONCTION, DEVRAIT PAS Y AVOIR AUTANT DE CONTACTS
        self.push_buttons_in_contact(contact_points)

        type_checked_tensor=self.check_type(id_tensor=contact_points)

        # Try to have the second dimension of size 6

        unique_type_tensor = torch.unique(type_checked_tensor,dim=-1)

        max_row_size = 6
        zeros_to_pad = max_row_size - unique_type_tensor.size(1)

        padded_tensor = F.pad(unique_type_tensor,(0,zeros_to_pad))
        current_state["contactPoints"] = padded_tensor

        room_tensor.check_buttons_pushed(self.state_tensor) # opens door if all buttons pushed

        self.add_to_memory_state(current_state)

        return current_state

    def init_memory_state(self):############### FINI ##################
        room_tensor = self.room_manager.room_array[self.actual_room]
        state={}
        state["CharacterPosition"] = torch.full((self.num_envs,3),math.nan)
        state["doorState"] = torch.full((self.num_envs,),math.nan)
        state["doorPosition"] = torch.full((self.num_envs,2),math.nan)
        state["buttonsState"] = torch.full((len(room_tensor.buttons_array_tensor),self.num_envs),math.nan)
        state["contactPoints"] = torch.full((self.num_envs,6),math.nan).to(torch.long)
        memory_state=[]
        for _ in range(5):
            memory_state.append(state)
        return memory_state

    def reset_memory_state(self,reset_tensor):###################### FINI ######################
        room_tensor = self.room_manager.room_array[self.actual_room]
        num_resets= torch.sum(reset_tensor).item()
        for i in range(5):
            self.memory_state[i]["CharacterPosition"][reset_tensor]=torch.full((num_resets,3),math.nan)
            self.memory_state[i]["doorState"][reset_tensor]=torch.full((num_resets,),math.nan)
            self.memory_state[i]["doorPosition"][reset_tensor]=torch.full((num_resets,2),math.nan)
            for j in range(len(room_tensor.buttons_array_tensor)):
                self.memory_state[i]["buttonsState"][j][reset_tensor]=torch.full((num_resets,),math.nan)
            
            self.memory_state[i]["contactPoints"][reset_tensor]=torch.full((num_resets,6),math.nan).to(torch.long)


    def flat_memory(self):  # met l'observation dans le bon format nécessaire à l'entrainement ############################### FINI ####################################
        obs = torch.cat((self.memory_observation[0],self.memory_observation[1],self.memory_observation[2],self.memory_observation[3],self.memory_observation[4]),dim=1)
        new_obs = torch.empty((self.num_envs,210))
        for i in range(5):
            new_obs[:, i*21:(i+1)*21] = obs[:,(2*i)*21:(2*i+1)*21]
            new_obs[:,105+ i * 21:105+(i + 1) * 21] = obs[:,(2 * i+1) * 21: (2 * i + 2) * 21]
        print("NEW OBSAAAAAAAAH : "+str(new_obs))
        return new_obs

    def get_contact_points(self):  #### sera fausse car albert n'est pas AABB ################################## FINI ############################
        room_tensor = self.room_manager.room_array[self.actual_room]
        id_collision_tensor = torch.full((self.num_envs,1), -1)
        albert_pos = self.get_pos_tensor()
        Amin_alb = albert_pos - 0.35
        Amax_alb = albert_pos + 0.35

        for button in room_tensor.buttons_array_tensor:
            button_pos = self.state_tensor[button.id_tensor][:,:3]
            Amin_button = button_pos - torch.tensor([0.5, 0.5, 0.1])
            Amax_button = button_pos + torch.tensor([0.5, 0.5, 0.1])
            result = check_collision_AABB_2(Amin_button, Amax_button, Amin_alb, Amax_alb)
            id_collision_tensor = torch.where(result.unsqueeze(1), torch.cat((id_collision_tensor, button.id_tensor.reshape(self.num_envs,1)), axis=1),torch.cat((id_collision_tensor, torch.full((self.num_envs,1),-1)), axis=1))

        for box_id in room_tensor.floor_array_tensor:
            box_pos = self.state_tensor[box_id.to(torch.int)][:,:3]
            Amin_box = box_pos - 0.5
            Amax_box = box_pos + 0.5
            result = check_collision_AABB_2(Amin_box, Amax_box, Amin_alb, Amax_alb)
            id_collision_tensor = torch.where(result.unsqueeze(1), torch.cat((id_collision_tensor, box_id.reshape(self.num_envs,1)), axis=1),torch.cat((id_collision_tensor, torch.full((self.num_envs,1),-1)), axis=1))

        for box_id in room_tensor.wall_array_tensor:
            box_pos = self.state_tensor[box_id.to(torch.int)][:,:3]
            Amin_box = box_pos - 0.5
            Amax_box = box_pos + 0.5
            result = check_collision_AABB_2(Amin_box, Amax_box, Amin_alb, Amax_alb)
            id_collision_tensor = torch.where(result.unsqueeze(1), torch.cat((id_collision_tensor, box_id.reshape(self.num_envs,1)), axis=1),torch.cat((id_collision_tensor, torch.full((self.num_envs,1),-1)), axis=1))

        # for door :
        door_pos = self.state_tensor[room_tensor.door_array_tensor.id_tensor][:,:3]
        Amin_box = door_pos - 0.5
        Amax_box = door_pos + 0.5
        result = check_collision_AABB_2(Amin_box, Amax_box, Amin_alb, Amax_alb)
        id_collision_tensor = torch.where(result.unsqueeze(1), torch.cat((id_collision_tensor, room_tensor.door_array_tensor.id_tensor.reshape(self.num_envs,1)), axis=1),torch.cat((id_collision_tensor, torch.full((self.num_envs,1),-1)), axis=1))

        id_collision_tensor=self.get_most_values(id_collision_tensor)# réduction à taille 6
        
        return id_collision_tensor

    def in_contact_with_floor_or_button(self):  # retourne true si albert est en contact avec le sol ou un boutton ########## FINI ##################s
        contact_points_tensor = self.get_contact_points()  ############### Cette fonction est aussi à changer
        types_checked_tensor = self.check_type(contact_points_tensor)

        bool_result = torch.tensor((types_checked_tensor == 1) | (types_checked_tensor == 2)).any(dim=1)
        return bool_result

    def get_pos_tensor(self):################ FINI ################
        positions = self.state_tensor[self.id_array][:,:3]
        return positions

    def get_ori_tensor(self):################### FINI #####################
        quats = self.state_tensor[self.id_array][:,3:7]
        return quats

    def push_buttons_in_contact(self,contact_points):################# FINI #####################
        # on va considérer que albert ne peut qu'appuyer sur un boutton à la fois ( ils sont assez espacés )
        room_tensor = self.room_manager.room_array[self.actual_room]
        type_checked_tensor = self.check_type(id_tensor=contact_points)
        bool_tensor1=type_checked_tensor == 1 # looking for types 1 : to change to enum values button
        button_detected_tensor = (bool_tensor1).nonzero() # retrieves the indices of rooms with button contact
        button_detected_ids = contact_points[button_detected_tensor[:,0],button_detected_tensor[:,1]] # gets the ids of detected buttons
        #T1 = torch.zeros((self.num_envs,6))
        #T1[bool_tensor2]=button_detected_ids # tenseur de taille num_envs avec des ids au meme indice que les true
        # PROBLEME DANS LE CAS OU LES 6 BOUTTONS SONT PAS APPUYÉS
        

        # this part is to add id=-1 in the environments where albert hasn't pushed any button during the step
        button_detected_ids_completed=torch.full((self.num_envs,),-1)
        indices = torch.arange(self.num_envs)
        mask = torch.isin(indices,button_detected_tensor[:,0])
        j=0
        for i in range(self.num_envs):
            if mask[i]:
                button_detected_ids_completed[i]=button_detected_ids[j]
                j+=1
        ##


        num_button = 0
        num_button_tensor = torch.full((self.num_envs,),-1)
        for button in room_tensor.buttons_array_tensor:
            num_button_tensor = torch.where((button.id_tensor==button_detected_ids),num_button,num_button_tensor)

            t2 = num_button_tensor==num_button # on choppe le bon indice
            buttons_to_push = t2 & ~button.is_pressed # true si bon indice et pas encore pressé
            button.got_pressed(self.state_tensor,buttons_to_push) # on appuie sur les bons bouttons

            num_button+=1

    def get_most_values(self,tensor):
        reduced_tensor=torch.full((self.num_envs,6),-1)
        for i in range(tensor.size(0)):
            row=tensor[i]
            non_minus1_indices = (row != -1).nonzero().view(-1)
            num_elements=min(6,non_minus1_indices.size(0))
            reduced_tensor[i,:num_elements] = row[non_minus1_indices[:num_elements]]
        return reduced_tensor







def binarize(buttons_tensor):  # retourne une liste d'états des bouttons ( 1 si le boutton à été appuyé dessus, 0 sinon ) ################################ FINI ######################
    list=[]
    for button_array in buttons_tensor:
        tensor_ = button_array.is_pressed.int()
        list.append(tensor_)
    stacked_tensor = torch.cat(list,dim=0).view(len(buttons_tensor),button_array.is_pressed.numel())
            
    return stacked_tensor


def euler_to_rotation_matrix(euler_angles):########################### FINI #########################
    """
    Convert a tensor of Euler angles to a tensor of rotation matrices.

    Args:
        euler_angles (torch.Tensor): Tensor of Euler angles in radians with shape (..., 3).

    Returns:
        torch.Tensor: Tensor of rotation matrices with shape (..., 3, 3).
    """
    roll, pitch, yaw = torch.unbind(euler_angles, dim=-1)

    # Calculate the individual rotation matrices
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    rotation_x = torch.stack([torch.ones_like(cos_r), torch.zeros_like(cos_r), torch.zeros_like(cos_r),
                              torch.zeros_like(cos_r), cos_r, -sin_r,
                              torch.zeros_like(cos_r), sin_r, cos_r], dim=-1).view(*euler_angles.shape[:-1], 3, 3)

    rotation_y = torch.stack([cos_p, torch.zeros_like(cos_p), sin_p,
                              torch.zeros_like(cos_p), torch.ones_like(cos_p), torch.zeros_like(cos_p),
                              -sin_p, torch.zeros_like(cos_p), cos_p], dim=-1).view(*euler_angles.shape[:-1], 3, 3)

    rotation_z = torch.stack([cos_y, -sin_y, torch.zeros_like(cos_y),
                              sin_y, cos_y, torch.zeros_like(cos_y),
                              torch.zeros_like(cos_y), torch.zeros_like(cos_y), torch.ones_like(cos_y)], dim=-1).view(
        *euler_angles.shape[:-1], 3, 3)

    # Combine the rotations to form the final rotation matrices
    rotation_matrices = torch.matmul(rotation_z, torch.matmul(rotation_y, rotation_x))

    return rotation_matrices


def quaternion_to_euler(quaternions):######################### FINI #########################
    """
    Convert a tensor of quaternions to a tensor of Euler angles in radians.

    Args:
        quaternions (torch.Tensor): Tensor of quaternions with shape (..., 4).

    Returns:
        torch.Tensor: Tensor of Euler angles in radians with shape (..., 3).
    """
    qw, qx, qy, qz = torch.unbind(quaternions, dim=-1)

    # Conversion to Euler angles
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    pitch = torch.asin(2 * (qw * qy - qz * qx))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))

    return torch.stack((roll, pitch, yaw), dim=-1)

def euler_to_quaternion(euler_angles):##################### FINI ######################
    """
    Convert a 1D tensor of XYZ Euler angles to a tensor of quaternions.

    Args:
        euler_angles (torch.Tensor): 1D tensor of Euler angles in radians with shape (3,).

    Returns:
        torch.Tensor: Tensor of quaternions with shape (4,).
    """
    roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]

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

    quaternion = torch.tensor([w, x, y, z])
    return quaternion




def check_collision_AABB(to_check_tensor,Amin,Amax,Bmin,Bmax):########################### FINI #############################
    condition = ((Amin<=Bmax) | (Amax>=Bmin))
    result = torch.all(condition,dim=1)
    return result & to_check_tensor

def check_collision_AABB_2(Amin,Amax,Bmin,Bmax):############################ FINI ##################################
    condition = ((Amin<=Bmax) | (Amax>=Bmin))
    result = torch.all(condition,dim=1)
    return result

