from typing import TypeVar, Generic
import numpy as np

# Assuming these imports match your project's structure
from .data_gen import DataGenerator
from feedback import PartialOrderFeedback
from pos_filter import PositionalFilter, DeterministicPositionalFilter
from ranking_models import RankingModel

class SyntheticPositionalFilteredDataGenerator(DataGenerator[PartialOrderFeedback]):
    """
    Generates data by sampling a full ranking from a ranking model 
    and applying a positional filter to generate feedback.
    """
    def __init__(self, n: int, ranking_model: RankingModel, pos_filter: PositionalFilter):
        super().__init__(n)
        self.ranking_model = ranking_model
        self.pos_filter = pos_filter

    def __next__(self) -> PartialOrderFeedback:
        # Sample a full ranking from the model
        # Expects an array where index = position, value = item_id
        full_ranking = self.ranking_model.sample()
        
        # Apply the filter to extract the partial order feedback
        return self.pos_filter.filter(full_ranking.tolist())


class SyntheticTopHDataGenerator(SyntheticPositionalFilteredDataGenerator):
    """
    Generates Top-h ranking feedback by passing a Top-h 
    DeterministicPositionalFilter to the base generator.
    """
    def __init__(self, n: int, h: int, ranking_model: RankingModel):
        if not 1 <= h < n:
            raise ValueError("h must be between 1 and n_items-1.")
        
        self.h = h

        # Generate the position pairs for Top-h.
        # An item in the top-h (position i) is preferred over ANY item 
        # that appears after it (position j).
        positions = []
        for i in range(h):
            for j in range(i + 1, n):
                positions.append((i, j))

        # Instantiate the deterministic filter we built earlier
        pos_filter = DeterministicPositionalFilter(n=n, positions=positions)
        
        # Initialize the base class with the model and the new filter
        super().__init__(n=n, ranking_model=ranking_model, pos_filter=pos_filter)
