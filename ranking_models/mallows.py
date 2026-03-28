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
        Returns an array where the index is the position, and the value is the item ID.
        """
        # 1. RIM generates a ranking centered around the identity [0, 1, ..., n-1]
        ranking: List[int] = [0]
        for i in range(1, self.n_items):
            idx_range = np.arange(i + 1)
            probs = self.phi ** (i - idx_range)
            probs /= np.sum(probs)
            
            pos = int(np.random.choice(idx_range, p=probs))
            ranking.insert(pos, i)
        
        # 2. ranking is currently a list of indices representing dispersion from the ideal.
        # Example: if ranking is [1, 0, 2], it means the 2nd best item is in 1st place.
        identity_dispersion = np.array(ranking, dtype=np.int64)
        
        # 3. Map the identity dispersion directly to your true_ranking items
        return self.true_ranking[identity_dispersion]
