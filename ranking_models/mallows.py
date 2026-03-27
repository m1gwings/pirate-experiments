from typing import List

from .ranking_model import RankingModel
import numpy as np

class MallowsRankingModel(RankingModel):
    """
    Mallows Model with Kendall Distance.
    Satisfies strict complete consensus and full support. [cite: 248, 362, 393]
    """
    def __init__(self, n_items: int, phi: float, true_ranking: List[int]) -> None:
        super().__init__(n_items)
        self.phi = phi
        self.true_ranking = np.array(true_ranking)

    def sample(self) -> np.ndarray:
        """
        Samples using the Repeated Insertion Method (RIM) for Kendall-Mallows.
        Returns a permutation where result[i] is the rank of item i. [cite: 174]
        """
        # RIM generates a ranking (list of items from best to worst)
        ranking: List[int] = [0]
        for i in range(1, self.n_items):
            # Probability of inserting item i at position j relative to current items
            # Based on Kendall distance dispersion [cite: 248, 250]
            idx_range = np.arange(i + 1)
            probs = self.phi ** (i - idx_range)
            probs /= np.sum(probs)
            
            pos = int(np.random.choice(idx_range, p=probs))
            ranking.insert(pos, i)
        
        # Convert ranking list [item_at_rank_0, item_at_rank_1, ...] 
        # to a permutation array pi where pi[item] = rank. [cite: 174]
        permutation = np.zeros(self.n_items, dtype=np.int64)
        for rank, item_idx in enumerate(ranking):
            permutation[item_idx] = rank
            
        return permutation