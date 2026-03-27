import numpy as np
from typing import List, Tuple, Any
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
import random

def get_experiment() -> Experiment:
    n: int = 20  
    num_samples: int = 5000  
    k: int = 10
    
    true_ranking: List[int] = list(range(n))

    phi: float = 0.8
    mallows = MallowsRankingModel(n_items=n, phi=phi, true_ranking=true_ranking)

    position_pairs: List[Tuple[int, int]] = [
        (r, r+1)
        for r in range(n // 2+1, n-1)
    ]

    pos_filter = DeterministicPositionalFilter(n=n, positions=position_pairs)

    baselines: List[Any] = [
        PirateAlgorithm(n=n),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
    ]
    
    return Experiment(
        name="Mallows MNAR",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=SyntheticPositionalFilteredDataGenerator(
            n=n, 
            ranking_model=mallows, 
            pos_filter=pos_filter
        ),
        baselines=baselines,
        loss=kendall_tau_loss,
        k=k
    )
