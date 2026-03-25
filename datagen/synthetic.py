import numpy as np
from typing import Optional
from .datagen import DataGenerator
from feedback import TopHRankingFeedback

class SyntheticTopHDataGenerator(DataGenerator[TopHRankingFeedback]):
    """
    Generates a Top-h ranking by sampling items without replacement 
    according to the Plackett-Luce model.
    """
    def __init__(self, n_items: int, h: int, theta: Optional[np.ndarray] = None):
        super().__init__(n_items)
        if not 1 <= h < n_items:
            raise ValueError("h must be between 1 and n_items-1.")
        self.h = h

        # Initialize item skill levels (theta)
        if theta is not None:
            if len(theta) != n_items:
                raise ValueError("theta must be an array of length n_items.")
            self.theta = np.array(theta)
        else:
            # Default to uniform skill levels if none are provided
            self.theta = np.zeros(n_items)
            
        # Pre-calculate base probabilities. 
        # Plackett-Luce weights are proportional to exp(theta)
        weights = np.exp(self.theta)
        self.base_probabilities = weights / np.sum(weights)

    def __next__(self) -> TopHRankingFeedback:
        # np.random.choice with replace=False and a probability distribution 
        # exactly mimics the Plackett-Luce sequential sampling process.
        top_h_ids = np.random.choice(
            self.n_items, 
            size=self.h, 
            replace=False, 
            p=self.base_probabilities
        ).tolist()
        
        # The remaining items become the 'others_ids'
        # We can use sets for faster lookup if n_items is very large
        top_h_set = set(top_h_ids)
        others_ids = [i for i in range(self.n_items) if i not in top_h_set]
        
        return TopHRankingFeedback(top_h_ids=top_h_ids, others_ids=others_ids)
