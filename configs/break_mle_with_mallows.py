import numpy as np
from typing import List, Tuple, Any, Dict, Optional, Sequence
from abc import ABC, abstractmethod

# Interfaces and imports based on your provided structure
from experiment.experiment import Experiment
from algos import PirateAlgorithm 
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
from algos import RankBreakableToPreferenceAdapter
from pos_filter import DeterministicPositionalFilter
from data_generator import SyntheticPositionalFilteredDataGenerator
from ranking_models import MallowsRankingModel

def get_experiment() -> Experiment:
    """
    Configures the experiment to demonstrate MLE failure under MNAR positional censoring. [cite: 110, 155]
    """
    n: int = 20  
    num_samples: int = 5000  
    k: int = 10
    delta_samples: int = int(num_samples / 300)
    
    # Ground Truth: Identity ranking [0, 1, 2, ..., 19]
    true_ranking: List[int] = list(range(n))

    # Mallows Model with high noise (phi=0.8) to allow for frequent inversions. [cite: 248]
    phi: float = 0.9
    mallows = MallowsRankingModel(n_items=n, phi=phi, true_ranking=true_ranking)

    # The "Inversion Trap" Positional Filter
    # Introduces Missing Not At Random (MNAR) comparisons. [cite: 110, 335]
    trap_positions: List[Tuple[int, int]] = []
    
    # 1. Reliable Adjacent Chain: PIRATE uses these to infer the true order via transitivity. [cite: 451]
    # Connects positions 0-1, 1-2, ..., 17-18
    for i in range(4, 8):
        trap_positions.append((i, i + 1))
        
    # 2. The Trap: Only observe item 19 vs item 0 when they are at opposite latent ranks.
    # Because of high noise (phi=0.8), item 19 will occasionally be at rank 0 (best) 
    # while item 0 is at rank 19 (worst). [cite: 661, 748]
    # The filter records THIS comparison, biasing the observed mu_19,0. 
    trap_positions.append((0, 19)) 

    pos_filter = DeterministicPositionalFilter(n=n, positions=trap_positions)

    # Baselines using the RankBreakable adapter for list-wise compatibility. [cite: 1069]
    baselines: List[Any] = [
        PirateAlgorithm(n=n), # Robust to MNAR through transitive closure. [cite: 444, 509]
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)), # Breaks as it assumes MCAR. [cite: 63, 1068]
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)), # Sensitive to positional bias. [cite: 1256]
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)), # Stationary distribution becomes biased. [cite: 1009, 1254]
    ]
    
    return Experiment(
        name="Mallows-Kendall MNAR Failure",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=SyntheticPositionalFilteredDataGenerator(
            n=n, 
            ranking_model=mallows, 
            pos_filter=pos_filter
        ),
        baselines=baselines,
        loss=kendall_tau_loss,
        k=k,
        delta_samples=delta_samples
    )