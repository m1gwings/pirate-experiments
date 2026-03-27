from feedback import PartialOrderFeedback
import numpy as np
from .algo import Algorithm

class PirateAlgorithm(Algorithm[PartialOrderFeedback]):
    def __init__(self, n: int) -> None:
        super().__init__(name="PIRATE")
        self.n = n
        
        self.w = np.zeros((n, n))
        self.E = np.zeros((n, n), dtype=bool)

    def feedback(self, feedback: PartialOrderFeedback) -> None:
        for pref in feedback.rank_breaking():
            i, j = pref.prec_id, pref.succ_id
            
            self.w[i, j] += 1
            N = self.w[i, j] + self.w[j, i]
            
            mu_ij = self.w[i, j] / N
            mu_ji = self.w[j, i] / N
            
            self.E[i, j] = mu_ij > mu_ji
            self.E[j, i] = mu_ji > mu_ij

    def predict(self) -> list[int]:
        # 1. Recompute transitive closure to get final out-degrees
        C = self.E.copy()
        for x in range(self.n):
            C |= C[:, x:x+1] & C[x:x+1, :]
            
        np.fill_diagonal(C, False)
        
        out_degrees = C.sum(axis=1)
        
        # 2. Sort descending and extract top-k
        # np.argsort sorts ascending, so we negate the scores to get descending        
        return np.argsort(-out_degrees).tolist()
