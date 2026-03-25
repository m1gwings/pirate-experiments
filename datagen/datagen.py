from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterator
from feedback import Feedback

F = TypeVar('F', bound=Feedback)

class DataGenerator(ABC, Generic[F], Iterator[F]):
    """Abstract base class for data generators that yield feedback samples."""
    def __init__(self, n_items: int) -> None:
        self.n_items = n_items

    @abstractmethod
    def __next__(self) -> F:
        """Returns the next feedback sample."""
        pass

    def __iter__(self) -> Iterator[F]:
        return self