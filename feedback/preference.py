from .feedback import Feedback
from item import Item

class PreferenceFeedback(Feedback):
    def __init__(self, prec: Item, succ: Item):
        if prec.id == succ.id:
            raise ValueError(f"Strict inequality violation: Item cannot precede itself (id: {prec.id})")
        
        self.prec = prec
        self.succ = succ
