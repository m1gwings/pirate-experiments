import numpy as np
from typing import Optional
from .data_gen import DataGenerator
from feedback import TopHRankingFeedback
from ranking_models.plackett_luce import PlackettLuce

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
            self.theta = np.ones(n_items)
            
        self.pl_model = PlackettLuce(self.theta)

    def __next__(self) -> TopHRankingFeedback:
        # Sample a full ranking from the Plackett-Luce model
        full_ranking = self.pl_model.sample()
        
        # Take the top h items
        top_h_ids = full_ranking[:self.h].tolist()
        
        # The remaining items become the 'others_ids'
        others_ids = full_ranking[self.h:].tolist()
        
        return TopHRankingFeedback(top_h_ids=top_h_ids, others_ids=others_ids)
