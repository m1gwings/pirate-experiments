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
from algos import Algorithm

def get_experiment() -> Experiment:
    print("Setting up Mallows MNAR experiment...")

    n: int = 20  
    num_samples: int = 10000  
    k: int = 20
    num_runs = 5
    
    true_ranking: List[int] = np.random.permutation(n).tolist()

    phi: float = 0.8
    mallows = MallowsRankingModel(n_items=n, phi=phi, true_ranking=true_ranking)

    def baselines_factory() -> List[Algorithm]: return [
        PirateAlgorithm(n=n),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
    ]
    
    return Experiment(
        name="Mallows MNAR",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generators=[SyntheticPositionalFilteredDataGenerator(
            n=n,
            ranking_model=mallows,
            pos_filter=DeterministicPositionalFilter(
                n=n,
                positions=[ (1, 2), (2, 3), (17, 18), (18, 19) ]
            )
        ) for _ in range(num_runs)],
        baselines_factory=baselines_factory,
        loss=kendall_tau_loss,
        k=k
    )
