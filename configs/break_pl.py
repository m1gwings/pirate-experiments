import numpy as np
from typing import List, Tuple

from experiment.experiment import Experiment
from algos import PirateAlgorithm 
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
from ranking_models.rum import RUM
from algos import RankBreakableToPreferenceAdapter, RankBreakableToPartialOrderAdapter
from pos_filter import DeterministicPositionalFilter
from data_generator import SyntheticPositionalFilteredDataGenerator

def get_experiment() -> Experiment:
    n = 20  
    num_samples = 50000  
    k = 10
    delta_samples = int(num_samples / 10)
    
    # 1. Non-linear Utility Spacing (The Noise Plateau)
    # We push the extreme items far apart to break PL, but compress 
    # the middle 16 items into a tiny utility band to starve the algorithms of signal.
    utilities = np.zeros(n)
    utilities[0] = 5.0
    utilities[1] = 4.9
    utilities[n-2] = -4.9
    utilities[n-1] = -5.0
    
    # The middle items have a gap of ~0.013 between them
    utilities[2:n-2] = np.linspace(0.1, -0.1, n-4)
    
    true_ranking = np.argsort(utilities)[::-1].tolist()

    # 2. Strictly Bounded Uniform Noise
    # Noise spans from -2 to 2 (max diff is 4). 
    # Extreme gaps (>4) are 100% deterministic (breaks PL). 
    # Middle gaps (~0.01) have a win rate of ~50.12%, making them incredibly noisy to learn.
    def bounded_uniform_noise(size: int) -> np.ndarray:
        return np.random.uniform(low=-2.0, high=2.0, size=size)

    rum = RUM(utilities=utilities, noise_distribution=bounded_uniform_noise)

    # 3. The Positional Filter
    base_positions: List[Tuple[int, int]] = []
    
    for i in range(n - 1):
        base_positions.append((i, i + 1))
        
    base_positions.append((0, n - 1))
    base_positions.append((1, n - 2))

    pos_filter = DeterministicPositionalFilter(n=n, positions=base_positions)

    # 4. Baselines
    baselines = [
        RankBreakableToPartialOrderAdapter(PirateAlgorithm(n=n)), 
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
    ]
    
    return Experiment(
        name="PL_MLE_Breaker_High_Noise_Middle",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=SyntheticPositionalFilteredDataGenerator(n=n, ranking_model=rum, pos_filter=pos_filter),
        baselines=baselines,
        loss=kendall_tau_loss,
        k=k,
        delta_samples=delta_samples
    )