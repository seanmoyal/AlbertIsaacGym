from Isaac.ObjetsEnvironnementTensor.ButtonTensor import Button
from Isaac.ObjetsEnvironnementTensor.DoorTensor import Door
import torch

class Room:  # classe d'une chambre ( niveau )
    def __init__(self,num_bodies,num_envs):

        self.num_envs=num_envs

        self.global_coord = torch.zeros((num_envs,3))  # l=0.5 # Coordonées globales de la chambre
        self.buttons_array_tensor = []
        self.floor_array_tensor = torch.tensor([])
        self.wall_array_tensor = torch.tensor([])
        self.iblocks_array_tensor = torch.tensor([])
        self.fences_array_tensor = torch.tensor([])
        self.door_array_tensor = None
        self.depth = 6
        self.width = 11
        self.height = 3
        self.num_bodies=num_bodies
        self.build_basic_room()

        # DEFINITION DU STATE : 

    def build_basic_room(self):  # ################################# FINI ##################################
        x, y, l = 0, 0, 0
        id = 1
        for i in range(self.depth):
            for j in range(self.width):
                self.floor_array_tensor = torch.cat((self.floor_array_tensor,torch.tensor([[id+self.num_bodies*env_id for env_id in range(self.num_envs)]])),dim=0)
                id+=1

                for z in range(self.height):  # MURS
                    if i == 0 or (j == 0 or j == 10):
                        if i == self.depth / 2 and (j == self.width - 1 or j == 0) and (z == 0):
                            if j == self.width - 1:
                                self.door_array_tensor = Door(torch.tensor([int(id+self.num_bodies*env_id) for env_id in range(self.num_envs)]))
                                id+=1
                        else:
                            self.wall_array_tensor = torch.cat((self.wall_array_tensor,torch.tensor([[id+self.num_bodies*env_id for env_id in range(self.num_envs)]])),dim=0)
                            id+=1

        button_id_tensor = torch.tensor([id+self.num_bodies*env_id for env_id in range(self.num_envs)])
        button_values = Button(button_id_tensor)
        self.buttons_array_tensor.append(button_values)



    def init_room(self, model, name='room1'): ################################ CHANGE TO ISAAC ####################################
        self.id = model.body(name).id
        id = self.id + 1
        while model.body(id).parentid[0] == self.id:
            body = model.body(id)
            name = body.name
            geom_id = body.geomadr[0]
            if "floor" in name:
                self.floor_array.append(id)
            elif "wall" in name:
                self.wall_array.append(id)
            elif "iblock" in name:
                self.iblocks_array.append(id)
            elif "door" in name:
                self.door_array = [id, Door(id)]
            elif "button" in name:
                self.buttons_array[id] = Button(id)
            elif "fence" in name:
                self.fences_array.append(id)
            id += 1

    def check_buttons_pushed(self, state_tensor):################################## FINI ############################
        open_door_tensor = self.door_array_tensor.is_opened
        buttons_all_pressed=torch.full((len(self.buttons_array_tensor),),True)
        for button_tensor in self.buttons_array_tensor:
            buttons_all_pressed = buttons_all_pressed & button_tensor.is_pressed

        all_buttons_pressed_but_door_closed = (buttons_all_pressed ^ open_door_tensor) & ~(open_door_tensor)
        self.door_array_tensor.open(state_tensor,all_buttons_pressed_but_door_closed)

    def reset_room(self,state_tensor,character_tensor,rooms_to_reset):  #################################### FINI #####################################
        # Evidemment elle est à compléter

        for button in self.buttons_array_tensor:
            # change in button pos
            condition = rooms_to_reset & button.is_pressed
            buttons_to_change_pos = torch.where(condition,button.id_tensor,-1)
            buttons_to_change_pos = torch.masked_select(buttons_to_change_pos,buttons_to_change_pos != -1)
            state_tensor[buttons_to_change_pos][:,2] = state_tensor[buttons_to_change_pos][:,2] + 1.9*0.2

            # resets to not pressed all buttons to reset
            button.is_pressed = torch.where(rooms_to_reset,False,button.is_pressed)

            # reset timing of character
            character_tensor.reset_time(rooms_to_reset)

            # closing doors
            self.door_array_tensor.close(state_tensor,rooms_to_reset)


    def get_values_button_from_button_array(self): ######################## FINI #################################
        id_values_button_tensor = torch.tensor([[self.buttons_array_tensor[i].keys(),self.buttons_array_tensor[i].values()] for i in range(self.num_envs)])
        split_tensors = torch.split(id_values_button_tensor,split_size_or_sections = 1,dim=2)
        return split_tensors[0],split_tensors[1]


