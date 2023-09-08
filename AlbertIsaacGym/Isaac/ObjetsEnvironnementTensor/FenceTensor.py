from Isaac.ObjetsEnvironnementTensor.CubeTensor import Cube
class Fence(Cube): # classe d'une barrière pour empêcher l'ia de passer

    def __init__(self,id_tensor,depth,height): #ajouter l'id
        self.id_tensor = id_tensor
        self.depth = depth
        self.height = height
