from experiment.experiment import Experiment
from algos.pirate import PiratePairwiseAdapterAlgorithm
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.bayes_mallows import BayesMallowsAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from feedback.preference import PreferenceFeedback
from loss.kendall_tau import kendall_tau_loss
import numpy as np

def get_experiment() -> Experiment:
    # Experiment parameters
    n = 10
    num_samples = 10000
    k = 5
    delta_samples = 50
    
    true_ranking = np.random.permutation(n).tolist()
    
    # Feedback generator
    def generator() -> PreferenceFeedback:
        # Simple generator: sample two items and return the one that is higher
        # in the true ranking.
        i, j = np.random.choice(n, 2, replace=False)
        pos_i = true_ranking.index(i)
        pos_j = true_ranking.index(j)
        if pos_i < pos_j:
            return PreferenceFeedback(prec_id=i, succ_id=j)
        else:
            return PreferenceFeedback(prec_id=j, succ_id=i)

    # Baselines
    baselines = [
        PiratePairwiseAdapterAlgorithm(n=n, delta=0.1),
        RankCentralityAlgorithm(n=n),
        BordaCountAlgorithm(n=n),
        BayesMallowsAlgorithm(n=n),
        PLPairwiseMLEAlgorithm(n=n)
    ]

    # Loss function
    loss = kendall_tau_loss
    
    return Experiment(
        name="Example Experiment",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=generator,
        baselines=baselines,
        loss=loss,
        k=k,
        delta_samples=delta_samples
    )
