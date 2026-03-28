import matplotlib
matplotlib.use('Agg')

import sys
from experiment.experiment import ExperimentBatch
from configs import mallows_mnar, sanity_check, movielens, mallows_mcar, plackett_luce_mcar, plackett_luce_mnar
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

    batch = ExperimentBatch(name="Experiment Batch", experiments=[
        # sanity_check.get_experiment(),
        # mallows_mnar.get_experiment(),
        # mallows_mcar.get_experiment(),
        plackett_luce_mnar.get_experiment(),
        plackett_luce_mcar.get_experiment(),
        movielens.get_experiment()
    ])
    batch.run()
    batch.plot()

if __name__ == '__main__':
    main()
