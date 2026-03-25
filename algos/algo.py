from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence
from feedback import Feedback

F = TypeVar('F', bound=Feedback)

class Result:
    # Using Sequence allows passing lists, tuples, or numpy arrays
    def __init__(self, top_k_ids: list[int]) -> None:
        if len(top_k_ids) != len(set(top_k_ids)):
            raise ValueError("All IDs in top_k_ids must be unique.")
            
        self.k: int = len(top_k_ids)
        self.top_k_ids = top_k_ids

class Algorithm(ABC, Generic[F]):
    def __init__(self, name: str) -> None:
        self.name = name
        self.sample_count: int = 0
    
    # Don't forget to type the feedback argument here too!
    def feed(self, feedback: F) -> None:
        self.sample_count += 1
        self._process_feedback(feedback)
    
    @abstractmethod
    def _process_feedback(self, feedback: F) -> None:
        pass

    @abstractmethod
    def can_predict(self) -> bool:
        pass

    @abstractmethod
    def predict(self) -> Result:
        pass
