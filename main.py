import matplotlib
matplotlib.use('Agg')

import sys
from experiment.experiment import ExperimentBatch
from configs import mallows_mnar, sanity_check
import rpy2.robjects as ro #type: ignore
import random
import numpy as np

def main():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    ro.r['set.seed'](seed)

    batch_1 = ExperimentBatch(name="Experiment Batch 1", experiments=[
        sanity_check.get_experiment(),
        mallows_mnar.get_experiment(),
        
    ])
    batch_1.run()
    batch_1.plot()

if __name__ == '__main__':
    main()
