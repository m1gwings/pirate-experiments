from typing import List, Dict, Callable, Any, Iterator
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from tqdm import tqdm  # type: ignore
import os
import csv
from datetime import datetime

# Assuming these have their own correct type hints in your project
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
                 num_recordings: int = 200,
                 initial_samples: int = 5) -> None:
        
        self.name: str = name
        self.num_samples: int = num_samples
        self.true_ranking: List[int] = true_ranking
        self.generator: Iterator[Feedback] = generator 
        self.baselines: Dict[str, Algorithm] = {algo.name: algo for algo in baselines}
        self.loss: Callable[[List[int], List[int], int], float] = loss
        self.k: int = k
        self.initial_samples: int = initial_samples
        
        # LOG-SPACING LOGIC:
        # Starts recording at 'initial_samples' and ends at 'num_samples'
        # This effectively skips plotting any data points before 'initial_samples'
        geom_points: Any = np.geomspace(initial_samples, num_samples, num=num_recordings)
        rounded_points: List[int] = [int(p) for p in np.round(geom_points)]
        
        # Ensure points are unique and within bounds
        self.recording_points: List[int] = sorted(list(set(
            p for p in rounded_points if initial_samples <= p <= num_samples
        )))
        
        self.results: Dict[str, List[float]] = {name: [] for name in self.baselines.keys()}
        self.sample_points: List[int] = []

    def _record_results(self, current_sample: int) -> None:
        """Records the loss for all algorithms at the current sample milestone."""
        self.sample_points.append(current_sample)
        for name, algo in self.baselines.items():
            pred_result: List[int] = algo.predict() 
            loss_val: float = self.loss(self.true_ranking, pred_result, self.k)
            self.results[name].append(loss_val)

    def run(self) -> None:
        """Executes the simulation and triggers recording at log-spaced milestones."""
        recording_idx: int = 0
        num_points: int = len(self.recording_points)
        
        for i in tqdm(range(1, self.num_samples + 1), desc=f"Running: {self.name}"):
            try:
                feedback: Feedback = next(self.generator)
            except StopIteration:
                break
            
            # Algorithms always see the feedback to update their models
            for algo in self.baselines.values():
                algo.feedback(feedback) 
            
            # Only record if we've reached a designated log-spaced point
            if recording_idx < num_points and i == self.recording_points[recording_idx]:
                self._record_results(i)
                recording_idx += 1

    def plot(self, ax: Axes) -> None:
        """Plots the results for all baselines without markers for a cleaner look."""
        for name, losses in self.results.items():
            # Marker removed for a smoother line-only aesthetic
            ax.semilogx(self.sample_points, losses, label=name, alpha=0.9, linewidth=2)
        
        ax.set_xlabel("Number of Samples (Log Scale)")
        ax.set_ylabel(f"Top-{self.k} Kendall Tau Loss")
        ax.set_title(self.name)
        ax.legend()
        
        # Grid lines help the eye follow the log-scale magnitudes
        ax.grid(True, which="both", ls="-", alpha=0.15)
        
        # Standard clean-plot formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

class ExperimentBatch:
    def __init__(self, name: str, experiments: List[Experiment]) -> None:
        self.name: str = name
        self.experiments: List[Experiment] = experiments

    def run(self) -> None:
        """Executes all experiments in the batch sequence."""
        for exp in self.experiments:
            exp.run()
    
    def plot(self) -> None:
        """Saves cleaned-up plots and CSV data to the output directory."""
        output_dir: str = "output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir_name: str = f"{self.name.lower().replace(' ', '_')}_{timestamp}"
        batch_output_dir: str = os.path.join(output_dir, batch_dir_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        for exp in self.experiments:
            plot_name: str = exp.name.lower().replace(" ", "_")
            
            # Explicit typing for fig/ax to satisfy MyPy
            fig: Any
            ax: Any
            fig, ax = plt.subplots(figsize=(10, 6))
            
            exp.plot(ax)
            plt.tight_layout()
            plt.savefig(os.path.join(batch_output_dir, f"{plot_name}.png"), dpi=300)
            plt.close(fig)
            
            # CSV export for external analysis
            csv_path: str = os.path.join(batch_output_dir, f"{plot_name}.csv")
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                baselines: List[str] = list(exp.results.keys())
                header: List[str] = ['Number of Samples'] + baselines
                writer.writerow(header)
                
                for i, sample_point in enumerate(exp.sample_points):
                    row: List[Any] = [sample_point]
                    for baseline in baselines:
                        row.append(exp.results[baseline][i])
                    writer.writerow(row)
