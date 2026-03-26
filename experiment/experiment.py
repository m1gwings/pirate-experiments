from typing import List, Dict, Callable, Any, Iterator
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from tqdm import tqdm  # type: ignore
import os
import csv
from datetime import datetime

# Assuming these have their own correct type hints
from algos.algo import Algorithm
from feedback.feedback import Feedback

class Experiment:
    def __init__(self,
                 name: str,
                 num_samples: int,
                 true_ranking: List[int],
                 generator: Iterator[Feedback],
                 baselines: List[Algorithm],
                 loss: Callable[[List[int], List[int], int], float],
                 k: int,
                 delta_samples: int) -> None:
        
        self.name: str = name
        self.num_samples: int = num_samples
        self.true_ranking: List[int] = true_ranking
        
        # CHANGED: Updated type hint to match Iterator
        self.generator: Iterator[Feedback] = generator 
        
        self.baselines: Dict[str, Algorithm] = {algo.name: algo for algo in baselines}
        
        self.loss: Callable[[List[int], List[int], int], float] = loss
        self.k: int = k
        self.delta_samples: int = delta_samples
        
        self.results: Dict[str, List[float]] = {name: [] for name in self.baselines.keys()}
        self.sample_points: List[int] = []

    def run(self) -> None:
        for i in tqdm(range(1, self.num_samples + 1), desc=f"Running Experiment: {self.name}", unit="sample"):
            
            # CHANGED: Call next() on the iterator instead of calling it like a function
            feedback: Feedback = next(self.generator)
            
            for algo in self.baselines.values():
                algo.feedback(feedback) 
            
            if i % self.delta_samples == 0:
                self.sample_points.append(i)
                for name, algo in self.baselines.items():
                    pred_result: List[int] = algo.predict() 
                    loss_val: float = self.loss(self.true_ranking, pred_result, self.k)
                    self.results[name].append(loss_val)

    def plot(self, ax: Axes) -> None:
        for name, losses in self.results.items():
            ax.plot(self.sample_points, losses, label=name)
            
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel(f"Top-{self.k} Kendall Tau Loss")
        ax.set_title(self.name)
        ax.legend()
        ax.grid(True)

class ExperimentBatch:
    def __init__(self, name: str, experiments: List[Experiment]) -> None:
        self.name: str = name
        self.experiments: List[Experiment] = experiments

    def run(self) -> None:
        for exp in self.experiments:
            exp.run()
    
    def plot(self) -> None:
        output_dir: str = "output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir_name: str = f"{self.name.lower().replace(' ', '_')}_{timestamp}"
        batch_output_dir: str = os.path.join(output_dir, batch_dir_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        for exp in self.experiments:
            plot_name: str = exp.name.lower().replace(" ", "_")
            
            # 1. Save the Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            exp.plot(ax)
            plt.savefig(os.path.join(batch_output_dir, f"{plot_name}.png"))
            plt.close(fig)
            
            # 2. Save the CSV Data
            csv_path: str = os.path.join(batch_output_dir, f"{plot_name}.csv")
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                
                baselines: List[str] = list(exp.results.keys())
                header: List[str] = ['Number of Samples'] + baselines
                writer.writerow(header)
                
                for i, sample_point in enumerate(exp.sample_points):
                    # Using List[Any] because it mixes ints and floats
                    row: List[Any] = [sample_point]
                    for baseline in baselines:
                        row.append(exp.results[baseline][i])
                    writer.writerow(row)
                    