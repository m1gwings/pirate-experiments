from experiment.experiment import Experiment
from algos import PirateAlgorithm
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.bayes_mallows import BayesMallowsAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
import numpy as np
from ranking_models.rum import RUM
from feedback.rank_breakable import RankBreakableFeedback
from feedback.preference import PreferenceFeedback
from algos.algo import Algorithm
from typing import TypeVar, Generic
from algos import RankBreakableToPartialOrderAdapter, RankBreakableToPreferenceAdapter
from data_generator import SyntheticTopHDataGenerator

def get_experiment() -> Experiment:
    # Experiment parameters
    n = 15
    h = 5
    num_samples = 5000
    k = 15
    delta_samples = 250
    
    # Generate utilities from a standard normal distribution
    utilities = np.random.normal(0, 1, n)
    true_ranking = np.argsort(utilities)[::-1].tolist()

    # Define reverse gumbel noise
    def reverse_gumbel_noise(size: int) -> np.ndarray:
        return -np.random.gumbel(size=size)

    # RUM with reverse gumbel noise
    rum = RUM(utilities=utilities, noise_distribution=reverse_gumbel_noise)

    # Baselines
    baselines = [
        RankBreakableToPartialOrderAdapter(PirateAlgorithm(n=n, delta=0.1)),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BayesMallowsAlgorithm(n=n)),
    ]

    # Loss function
    loss = kendall_tau_loss
    
    return Experiment(
        name="Trick PL with Reverse Gumbel RUM and Top-H feedback",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=SyntheticTopHDataGenerator(n=rum.n_items, ranking_model=rum, h=h),
        baselines=baselines,
        loss=loss,
        k=k,
        delta_samples=delta_samples
    )
