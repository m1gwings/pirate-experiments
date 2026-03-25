from abc import ABC

class Item(ABC):
    def __init__(self, id: str):
        self.id = id
