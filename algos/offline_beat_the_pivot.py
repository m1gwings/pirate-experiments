import math
import numpy as np
from .algo import Algorithm, Result
from feedback import TopHRankingFeedback

class OfflineBeatThePivotAlgorithm(Algorithm[TopHRankingFeedback]):
    def __init__(self, n: int, h: int, epsilon: float, delta: float) -> None:
        super().__init__(name="Offline Beat-the-Pivot")
        self.n = n
        self.h = h
        self.epsilon = epsilon
        self.delta = delta
        
        # Calculate theoretical sample complexity based on Algorithm 5 of Active Ranking frmo Subset-wise Preferences
        # Since we pull all arms offline, the subset size k is equal to n
        epsilon_prime = self.epsilon / 16.0
        delta_prime = self.delta / (8.0 * self.n)
        
        # t = (2k / (m * epsilon'^2)) * log(1 / delta')
        self.required_samples = math.ceil(
            (2.0 * self.n) / (self.h * (epsilon_prime ** 2)) * math.log(1.0 / delta_prime)
        )
        
        # w[i, j] stores the number of times item i beat item j 
        self.w = np.zeros((n, n))

    def _process_feedback(self, feedback: TopHRankingFeedback) -> None:
        for pref in feedback.rank_breaking():
            i, j = pref.prec_id, pref.succ_id
            self.w[i, j] += 1

    def can_predict(self) -> bool:
        # Strictly enforce the theoretical sample complexity
        return self.sample_count >= self.required_samples

    def predict(self) -> Result:
        if not self.can_predict():
            raise ValueError(
                f"Cannot predict yet. The algorithm requires {self.required_samples} "
                f"samples, but only has {self.sample_count}."
            )

        # 1. Find the pivot item 'b' via the empirical best overall
        total_wins = np.sum(self.w, axis=1)
        b = int(np.argmax(total_wins))
        
        # 2. Estimate the pivot-preference scores: p_{ib}
        p_ib = np.zeros(self.n)
        for i in range(self.n):
            if i == b:
                # The pivot is naturally the best item in this relative comparison
                p_ib[i] = float('inf') 
            else:
                den = self.w[i, b] + self.w[b, i]
                # If they never directly competed, default to a 0.5 tie-breaker
                p_ib[i] = self.w[i, b] / den if den > 0 else 0.5
                
        # 3. Sort descending to get the full ranking
        ranked_ids = np.argsort(p_ib)[::-1].tolist()
        
        # 4. Return the full ranking
        return Result(top_k_ids=ranked_ids)
