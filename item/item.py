from abc import ABC

class Item(ABC):
    def __init__(self, id: int) -> None:
        self.id = id
