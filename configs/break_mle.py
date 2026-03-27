import numpy as np
from typing import List, Tuple, Any
from collections import defaultdict

from experiment.experiment import Experiment
from algos import PirateAlgorithm 
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
from ranking_models.rum import RUM
from algos import RankBreakableToPreferenceAdapter
from pos_filter import DeterministicPositionalFilter
from data_generator import SyntheticPositionalFilteredDataGenerator
from feedback import PartialOrderFeedback, PreferenceFeedback

def get_experiment() -> Experiment:
    n = 20  
    num_samples = 50000  
    k = 10
    delta_samples = int(num_samples / 100)
    
    # Standard Linear Utilities
    utilities = np.random.uniform(0, 1, n)
    true_ranking = np.argsort(utilities)[::-1].tolist()

    def noise(size: int) -> np.ndarray:
        return np.random.random(size=size)

    rum = RUM(utilities=utilities, noise_distribution=noise)

    # The Middle-Exclusion Positional Trap
    base_positions: List[Tuple[int, int]] = []
    
    for i in range(4):
        base_positions.append((i, i + 1))
        
    for i in range(15, 19):
        base_positions.append((i, i + 1))
        
    base_positions.append((4, 15))

    pos_filter = DeterministicPositionalFilter(n=n, positions=base_positions)

    # Baselines with explicit typing to fix MyPy
    baselines: List[Any] = [
        PirateAlgorithm(n=n), 
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
    ]
    
    return Experiment(
        name="Break MLE",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=SyntheticPositionalFilteredDataGenerator(n=n, ranking_model=rum, pos_filter=pos_filter),
        baselines=baselines,
        loss=kendall_tau_loss,
        k=k,
        delta_samples=delta_samples
    )
