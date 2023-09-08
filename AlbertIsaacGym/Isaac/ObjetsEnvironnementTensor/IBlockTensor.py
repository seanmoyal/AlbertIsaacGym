
class IBlock(): # blocs les uns au dessus des autres, fait pour que l'ia apprenne à sauter dessus pour passer des obstacles par exemple
    def __init__(self,id_tensor,height=1):
        self.id=id_tensor
        self.height=height
