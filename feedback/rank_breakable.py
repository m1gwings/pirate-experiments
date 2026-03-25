from .preference import PreferenceFeedback
from abc import ABC, abstractmethod
from .feedback import Feedback

class RankBreakableFeedback(Feedback):
    @abstractmethod
    def rank_breaking(self) -> list[PreferenceFeedback]:
        pass
