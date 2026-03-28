import matplotlib
matplotlib.use('Agg')

import sys
from experiment.experiment import ExperimentBatch
from configs import mallows_mnar, sanity_check, movielens
import rpy2.robjects as ro #type: ignore
import random
import numpy as np
import os

def main():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    ro.r['set.seed'](seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    batch_1 = ExperimentBatch(name="Experiment Batch", experiments=[
        sanity_check.get_experiment(),
        # mallows_mnar.get_experiment(),
        # movielens.get_experiment()
    ])
    batch_1.run()
    batch_1.plot()

if __name__ == '__main__':
    main()
