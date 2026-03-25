from .algo import Algorithm, Result
from feedback import PreferenceFeedback
import numpy as np

class BordaCountAlgorithm(Algorithm[PreferenceFeedback]):
    def __init__(self, n: int, k: int) -> None:
        super().__init__(name="BordaCount")
        self.n = n
        self.k = k
        
        # Track wins and total comparisons per item
        self.wins = np.zeros(n, dtype=float)
        self.comparisons = np.zeros(n, dtype=float)

    def feedback(self, feedback: PreferenceFeedback) -> None:
        # prec_id is the preferred item (winner), succ_id is the less preferred item (loser)
        winner = feedback.prec_id
        loser = feedback.succ_id
        
        self.wins[winner] += 1
        self.comparisons[winner] += 1
        self.comparisons[loser] += 1

    def can_predict(self) -> bool:
        return bool(np.sum(self.comparisons) > 0)

    def predict(self) -> Result:
        scores = np.zeros(self.n, dtype=float)
        
        # Safely calculate win ratios, avoiding division by zero for uncompared items
        mask = self.comparisons > 0
        scores[mask] = self.wins[mask] / self.comparisons[mask]
        
        # Sort items by their Borda score in descending order
        ranked_indices = np.argsort(scores)[::-1]
        top_k_ids = [int(idx) for idx in ranked_indices[:self.k]]
                
        return Result(top_k_ids=top_k_ids)
