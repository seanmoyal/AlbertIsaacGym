

class RoomManager: # classe permettant de gérer les différents niveaux : les aligner, les emboiter(à venir), et à voir quoi rajouter

    def __init__(self):
        self.room_array=[]

    def add_room(self,room):# ajoute un niveau à sa liste de niveaux
        self.room_array.append(room)

