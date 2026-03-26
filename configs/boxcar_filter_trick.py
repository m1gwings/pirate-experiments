import numpy as np
from scipy.stats import gennorm # type: ignore
from typing import List, Tuple
from collections import defaultdict

from experiment.experiment import Experiment
from algos import PirateAlgorithm
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.bayes_mallows import BayesMallowsAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from loss.kendall_tau import kendall_tau_loss
from ranking_models.rum import RUM
from feedback.partial_order import PartialOrderFeedback
from feedback.preference import PreferenceFeedback
from algos import RankBreakableToPartialOrderAdapter, RankBreakableToPreferenceAdapter
from pos_filter import PositionalFilter, DeterministicPositionalFilter
from data_generator import SyntheticPositionalFilteredDataGenerator

# ---------------------------------------------------------
# 3. The Experiment Definition
# ---------------------------------------------------------
def get_experiment() -> Experiment:
    n = 50
    num_samples = 20000
    k = 20
    delta_samples = 500
    
    # 1. Generate latent utilities
    utilities = np.random.normal(0, 1, n)
    true_ranking = np.argsort(utilities)[::-1].tolist()

    # 2. RUM with Platykurtic "Boxcar" Noise (Generalized Normal)
    # beta=15 creates an extreme flat top with near-vertical drop-offs,
    # entirely destroying PL's logarithmic extreme-value assumptions.
    def platykurtic_boxcar_noise(size: int) -> np.ndarray:
        return gennorm.rvs(beta=15, loc=0, scale=1, size=size)

    rum = RUM(utilities=utilities, noise_distribution=platykurtic_boxcar_noise)

    # 3. Extreme MNAR Positional Filter
    # We feed the algorithms ONLY the deterministic extreme interactions.
    trick_positions = [
        (0, n-2), 
        (0, n-1), 
        (1, n-1),  # Added to further lock down the bottom tail
        (n-2, n-1)
    ]
    pos_filter = DeterministicPositionalFilter(n=n, positions=trick_positions)

    # 4. Baselines
    baselines = [
        RankBreakableToPartialOrderAdapter(PirateAlgorithm(n=n, delta=0.1)),
        RankBreakableToPreferenceAdapter(PLPairwiseMLEAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BordaCountAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(RankCentralityAlgorithm(n=n)),
        RankBreakableToPreferenceAdapter(BayesMallowsAlgorithm(n=n)),
    ]
    
    return Experiment(
        name="Annihilate PL and RC: High-Beta GenNorm + Overflows + MNAR",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=SyntheticPositionalFilteredDataGenerator(n=n, ranking_model=rum, pos_filter=pos_filter),
        baselines=baselines,
        loss=kendall_tau_loss,
        k=k,
        delta_samples=delta_samples
    )
