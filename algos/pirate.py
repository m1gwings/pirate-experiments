from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence
from feedback import *
import numpy as np
from .algo import *

class PirateAlgorithm(Algorithm[PartialOrderFeedback]):
    def __init__(self, T: int, n: int, delta: float) -> None:
        super().__init__(name="PIRATE")
        self.T = T
        self.n = n
        self.delta = delta
        
        self.w = np.zeros((n, n))
        self.E = np.zeros((n, n), dtype=bool)
        self.c = np.log(3 * self.T * self.n / self.delta)

    def feedback(self, feedback: PartialOrderFeedback) -> None:
        for pref in feedback.rank_breaking():
            i, j = pref.prec_id, pref.succ_id
            
            self.w[i, j] += 1
            N = self.w[i, j] + self.w[j, i]
            
            mu_ij = self.w[i, j] / N
            mu_ji = self.w[j, i] / N
            b_delta = np.sqrt(self.c / N)
            
            self.E[i, j] = (mu_ij - b_delta) > (mu_ji + b_delta)
            self.E[j, i] = (mu_ji - b_delta) > (mu_ij + b_delta)

    def predict(self) -> Result:
        # 1. Recompute transitive closure to get final out-degrees
        C = self.E.copy()
        for x in range(self.n):
            C |= C[:, x:x+1] & C[x:x+1, :]
            
        np.fill_diagonal(C, False)
        
        # Convert to float so we can add fractional noise
        out_degrees = C.sum(axis=1).astype(float)
        
        # 2. Add uniform random noise in [0, 0.5) to break ties.
        # Since the true out-degrees are integers, adding noise < 1 strictly 
        # preserves the valid DAG topological order while shuffling ties.
        noise = np.random.uniform(0, 0.5, size=self.n)
        noisy_degrees = out_degrees + noise
        
        # 3. Sort descending and extract top-k
        # np.argsort sorts ascending, so we negate the scores to get descending
        top_k_ids = np.argsort(-noisy_degrees).tolist()
        
        return Result(top_k_ids=top_k_ids)
    