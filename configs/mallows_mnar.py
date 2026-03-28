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
from data_generator import SyntheticPositionalFilteredDataGenerator, SyntheticTopHDataGenerator, SyntheticBottomHDataGenerator
from ranking_models import MallowsRankingModel
import random

def get_experiment() -> Experiment:
    n: int = 20  
    num_samples: int = 5000  
    k: int = 10
    
    true_ranking: List[int] = np.random.permutation(n).tolist()

    phi: float = 0.9
    mallows = MallowsRankingModel(n_items=n, phi=phi, true_ranking=true_ranking)

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
        generator=SyntheticBottomHDataGenerator(
            n=n,
            h=10,
            ranking_model=mallows, 
        ),
        baselines=baselines,
        loss=kendall_tau_loss,
        k=k
    )
