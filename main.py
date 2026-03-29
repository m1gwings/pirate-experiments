import matplotlib
matplotlib.use('Agg')

import sys
from experiment.experiment import ExperimentBatch
from configs import movielens, plackett_luce_mcar, plackett_luce_mnar, sushi_mnar
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

    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_fc7e9f87f05291ef1887c8af8249586e'

    batch = ExperimentBatch(name="Experiment Batch", experiments=[
        plackett_luce_mcar.get_experiment(),
        plackett_luce_mnar.get_experiment(),
        movielens.get_experiment(),
        sushi_mnar.get_experiment(),
    ])
    batch.run()
    batch.plot()

if __name__ == '__main__':
    main()
