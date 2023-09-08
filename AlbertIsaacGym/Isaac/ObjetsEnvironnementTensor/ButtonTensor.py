from Isaac.ObjetsEnvironnementTensor.CubeTensor import Cube
import torch

class Button(Cube):  # Classe des bouttons : spots verts ou l'acteur doit passer pour ouvrir la porte

    def __init__(self, id_tensor):
        self.is_pressed = torch.full((id_tensor.numel(),),False)  # variable d'état du boutton : faux si l'acteur n'est pas passée dessus
        self.id_tensor = id_tensor

    def got_pressed(self, state_tensor,got_pressed_tensor):  ################## FINI ###########################
        got_pressed_ids = torch.where(got_pressed_tensor,self.id_tensor,-1)
        got_pressed_ids = torch.masked_select(got_pressed_ids,got_pressed_ids != -1)
        state_tensor[got_pressed_ids][:,2] = state_tensor[got_pressed_ids][:,2] - 1.9 * 0.02  # supposing 0.02 is the button's size
        self.is_pressed = torch.where(got_pressed_tensor,True,self.is_pressed)

