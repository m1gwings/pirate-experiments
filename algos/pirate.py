from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence
from feedback import *
import numpy as np
from .algo import *

class PirateAlgorithm(Algorithm[PartialOrderFeedback]):
    def __init__(self, T: int, n: int, k: int, delta: float) -> None:
        super().__init__(name="PIRATE")
        self.T = T
        self.n = n
        self.k = k
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

    def can_predict(self) -> bool:
        C = self.E.copy()
        for x in range(self.n):
            C |= C[:, x:x+1] & C[x:x+1, :]
            
        np.fill_diagonal(C, False) 
        out_degrees = C.sum(axis=1)
        
        required_degrees = set(range(self.n - self.k, self.n))
        actual_degrees = set(out_degrees)
        
        return required_degrees.issubset(actual_degrees)

    def predict(self) -> Result:
        # 1. Recompute transitive closure to get final out-degrees
        C = self.E.copy()
        for x in range(self.n):
            C |= C[:, x:x+1] & C[x:x+1, :]
            
        np.fill_diagonal(C, False)
        out_degrees = C.sum(axis=1)
        
        # 2. Extract the top-k items in strict order
        top_k_ids = []
        
        # Iterate from n-1 (best item) down to n-k (k-th best item)
        for target_degree in range(self.n - 1, self.n - self.k - 1, -1):
            
            # np.where returns a tuple, so we take the first element [0]
            matches = np.where(out_degrees == target_degree)[0]
            
            if len(matches) > 0:
                # If can_predict() is True, there will be exactly one match per target_degree
                top_k_ids.append(int(matches[0]))
            else:
                # Safety net in case predict() is called before can_predict() resolves to True
                raise ValueError(
                    f"Cannot uniquely identify item with out-degree {target_degree}. "
                    "Make sure to call can_predict() first to guarantee top-k convergence."
                )
                
        return Result(top_k_ids=top_k_ids)
    