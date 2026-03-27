from experiment.experiment import Experiment
from algos import PirateAlgorithm, PreferenceToPartialOrderAdapter
from algos.rank_centrality import RankCentralityAlgorithm
from algos.borda_count import BordaCountAlgorithm
from algos.pl_pairwise_mle import PLPairwiseMLEAlgorithm
from feedback.preference import PreferenceFeedback
from loss.kendall_tau import kendall_tau_loss
import numpy as np
from typing import Iterator

def get_experiment() -> Experiment:
    # Experiment parameters
    n = 20
    num_samples = 10000
    k = 20
    
    true_ranking = np.random.permutation(n).tolist()
    
    # 1. Update the function to yield values infinitely
    def create_generator() -> Iterator[PreferenceFeedback]:
        while True:
            # Simple generator: sample two items and return the one that is higher
            # in the true ranking.
            i, j = np.random.choice(n, 2, replace=False)
            pos_i = true_ranking.index(i)
            pos_j = true_ranking.index(j)
            
            if pos_i < pos_j:
                yield PreferenceFeedback(prec_id=i, succ_id=j)
            else:
                yield PreferenceFeedback(prec_id=j, succ_id=i)

    # 2. Instantiate the iterator
    generator_iterator = create_generator()

    # Baselines
    baselines = [
        PreferenceToPartialOrderAdapter(PirateAlgorithm(n=n)),
        RankCentralityAlgorithm(n=n),
        BordaCountAlgorithm(n=n),
        PLPairwiseMLEAlgorithm(n=n)
    ]

    # Loss function
    loss = kendall_tau_loss
    
    return Experiment(
        name="Sanity Check",
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=generator_iterator, # 3. Pass the iterator instance
        baselines=baselines,
        loss=loss,
        k=k
    )
