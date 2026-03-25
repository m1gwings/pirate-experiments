from abc import ABC, abstractmethod
from .preference import PreferenceFeedback

class Feedback(ABC):
    pass

class RankBreakableFeedback(Feedback):
    @abstractmethod
    def rank_breaking(self) -> list[PreferenceFeedback]:
        pass
