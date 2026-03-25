from typing import List, Dict, Callable
import numpy as np
import matplotlib.pyplot as plt

from algos.algo import Algorithm, Result
from feedback.feedback import Feedback
from loss.kendall_tau import kendall_tau_loss

class Experiment:
    def __init__(self,
                 num_samples: int,
                 true_ranking: List[int],
                 generator: Callable[[], Feedback],
                 baselines: Dict[str, Algorithm],
                 loss: Callable[[List[int], List[int], int], float],
                 k: int,
                 delta_samples: int):
        self.num_samples = num_samples
        self.true_ranking = true_ranking
        self.generator = generator
        self.baselines = baselines
        self.loss = loss
        self.k = k
        self.delta_samples = delta_samples
        
        self.results: Dict[str, List[float]] = {name: [] for name in self.baselines.keys()}
        self.sample_points: List[int] = []

    def run(self) -> None:
        for i in range(1, self.num_samples + 1):
            feedback = self.generator()
            
            for algo in self.baselines.values():
                # MyPy is not smart enough to understand that the type of feedback
                # matches the type expected by the algorithm.
                # We can ignore the type check here.
                algo.feedback(feedback) # type: ignore
            
            if i % self.delta_samples == 0:
                self.sample_points.append(i)
                for name, algo in self.baselines.items():
                    if algo.can_predict():
                        pred_result: Result = algo.predict()
                        l = self.loss(self.true_ranking, pred_result.top_k_ids, self.k)
                        self.results[name].append(l)
                    else:
                        self.results[name].append(1.0) # Max loss if cannot predict
        
        self.plot()

    def plot(self) -> None:
        plt.figure(figsize=(10, 6))
        
        for name, losses in self.results.items():
            plt.plot(self.sample_points, losses, label=name)
            
        plt.xlabel("Number of Samples")
        plt.ylabel(f"Top-{self.k} Kendall Tau Loss")
        plt.title("Ranking Algorithm Performance")
        plt.legend()
        plt.grid(True)
        plt.show()

