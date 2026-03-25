import numpy as np
from algos.algo import Algorithm, Result
from algos.pirate import PirateAlgorithm
from algos.rank_centrality import RankCentralityAlgorithm
from experiment.experiment import Experiment
from feedback.preference import PreferenceFeedback
from feedback.partial_order import PartialOrderFeedback
from loss.kendall_tau import kendall_tau_loss
from typing import Type

class PirateAdapter(Algorithm[PreferenceFeedback]):
    def __init__(self, T: int, n: int, delta: float):
        super().__init__(name="PIRATE")
        self.pirate = PirateAlgorithm(T=T, n=n, delta=delta)
    
    def feedback(self, feedback: PreferenceFeedback) -> None:
        # PirateAlgorithm expects a PartialOrderFeedback, so we wrap the preference.
        self.pirate.feedback(PartialOrderFeedback(preferences=[feedback]))

    def predict(self) -> Result:
        return self.pirate.predict()

    def can_predict(self) -> bool:
        return self.pirate.can_predict()

def main():
    # Experiment parameters
    n = 10
    num_samples = 1000
    k = 5
    delta_samples = 100
    
    true_ranking = np.random.permutation(n).tolist()
    
    # Feedback generator
    def generator() -> PreferenceFeedback:
        # Simple generator: sample two items and return the one that is higher
        # in the true ranking.
        i, j = np.random.choice(n, 2, replace=False)
        pos_i = true_ranking.index(i)
        pos_j = true_ranking.index(j)
        if pos_i < pos_j:
            return PreferenceFeedback(succ_id=i, prec_id=j)
        else:
            return PreferenceFeedback(succ_id=j, prec_id=i)

    # Baselines
    baselines = {
        "PIRATE": PirateAdapter(T=num_samples, n=n, delta=0.1),
        "RankCentrality": RankCentralityAlgorithm(num_items=n)
    }

    # Loss function
    loss = kendall_tau_loss
    
    # Create and run experiment
    exp = Experiment(
        num_samples=num_samples,
        true_ranking=true_ranking,
        generator=generator,
        baselines=baselines,
        loss=loss,
        k=k,
        delta_samples=delta_samples
    )
    exp.run()

if __name__ == '__main__':
    main()
