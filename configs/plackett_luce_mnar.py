import numpy as np
from typing import List, Tuple, Any
from experiment.experiment import Experiment
from algos import PirateAlgorithm, BayesMallowsAlgorithm
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
from algos import RankBreakableToPreferenceAdapter
from pos_filter import DeterministicPositionalFilter, RandomPairPositionalFilter
from data_generator import SyntheticPositionalFilteredDataGenerator, SyntheticTopHDataGenerator, SyntheticBottomHDataGenerator
from ranking_models import MallowsRankingModel, PlackettLuceRankingModel
import random
from algos import Algorithm

def get_experiment() -> Experiment:
    print("Setting up Plackett-Luce MNAR experiment...")

    n: int = 15  
    num_samples: int = 10000000  
    k: int = 15
    num_runs = 5
    
    pl_parameters = np.random.uniform(0, 1, size=n)
    true_ranking: List[int] = np.argsort(pl_parameters).tolist()[::-1]

    pl = PlackettLuceRankingModel(pl_params=pl_parameters)

    def baselines_factory() -> List[Algorithm]: return [
        PirateAlgorithm(n=n),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
    ]
    
    return Experiment(
        name="Plackett-Luce MNAR",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generators=[SyntheticPositionalFilteredDataGenerator(
            n=n,
            ranking_model=pl,
            pos_filter=DeterministicPositionalFilter(
                n=n,
                positions=[ (1, 2), (2, 3), (n-3, n-2), (n-2, n-1) ]
            )
        ) for _ in range(num_runs)],
        baselines_factory=baselines_factory,
        loss=kendall_tau_loss,
        k=k
    )
