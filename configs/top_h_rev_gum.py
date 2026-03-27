from experiment.experiment import Experiment
from algos import PirateAlgorithm
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
import numpy as np
from ranking_models.rum import RUM
from algos import RankBreakableToPartialOrderAdapter, RankBreakableToPreferenceAdapter
from data_generator import SyntheticTopHDataGenerator

def get_experiment() -> Experiment:
    # Experiment parameters
    n = 15
    h = 5
    num_samples = 10000
    k = 15
    delta_samples = int(num_samples / 100)

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
        RankBreakableToPartialOrderAdapter(PirateAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
    ]

    # Loss function
    loss = kendall_tau_loss
    
    return Experiment(
        name="Top-H Feedback with Reverse Gumbel",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=SyntheticTopHDataGenerator(n=rum.n_items, ranking_model=rum, h=h),
        baselines=baselines,
        loss=loss,
        k=k,
        delta_samples=delta_samples
    )
