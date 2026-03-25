from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from feedback import Feedback

F = TypeVar('F', bound=Feedback)

class Algorithm(ABC, Generic[F]):
    def __init__(self, name: str) -> None:
        self.name = name
        self.sample_count: int = 0
    
    def feed(self, feedback) -> None:
        self.sample_count += 1
        self._process_feedback(feedback)
    
    @abstractmethod
    def _process_feedback(self, feedback: F) -> None:
        pass
