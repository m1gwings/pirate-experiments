from typing import List, Dict, Callable, Any
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # type: ignore
import os
from datetime import datetime

from algos.algo import Algorithm
from feedback.feedback import Feedback

class Experiment:
    def __init__(self,
                 name: str,
                 num_samples: int,
                 true_ranking: List[int],
                 generator: Callable[[], Feedback],
                 baselines: List[Algorithm],
                 loss: Callable[[List[int], List[int], int], float],
                 k: int,
                 delta_samples: int):
        self.name = name
        self.num_samples = num_samples
        self.true_ranking = true_ranking
        self.generator = generator
        self.baselines = {algo.name: algo for algo in baselines}
        self.loss = loss
        self.k = k
        self.delta_samples = delta_samples
        
        self.results: Dict[str, List[float]] = {name: [] for name in self.baselines.keys()}
        self.sample_points: List[int] = []

    def run(self) -> None:
        for i in tqdm(range(1, self.num_samples + 1), desc=f"Running Experiment: {self.name}", unit="sample"):
            feedback = self.generator()
            
            for algo in self.baselines.values():
                algo.feedback(feedback) # type: ignore
            
            if i % self.delta_samples == 0:
                self.sample_points.append(i)
                for name, algo in self.baselines.items():
                    pred_result: list[int] = algo.predict()
                    l = self.loss(self.true_ranking, pred_result, self.k)
                    self.results[name].append(l)

    def plot(self, ax: Any) -> None:
        for name, losses in self.results.items():
            ax.plot(self.sample_points, losses, label=name)
            
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel(f"Top-{self.k} Kendall Tau Loss")
        ax.set_title(self.name)
        ax.legend()
        ax.grid(True)

class ExperimentBatch:
    def __init__(self, name: str, experiments: List[Experiment]):
        self.name = name
        self.experiments = experiments

    def run(self) -> None:
        for exp in self.experiments:
            exp.run()
    
    def plot(self) -> None:
        # Create output directory
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Create batch-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir_name = f"{self.name.lower().replace(' ', '_')}_{timestamp}"
        batch_output_dir = os.path.join(output_dir, batch_dir_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        # Plot and save individual experiments
        for exp in self.experiments:
            fig, ax = plt.subplots(figsize=(10, 6))
            exp.plot(ax)
            plot_name = exp.name.lower().replace(" ", "_")
            plt.savefig(os.path.join(batch_output_dir, f"{plot_name}.png"))
            plt.close(fig)
        
