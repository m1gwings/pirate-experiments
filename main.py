from experiment.experiment import ExperimentBatch
from configs import sanity_check

def main():
    batch = ExperimentBatch(name="Experiment Batch 1", experiments=[
        sanity_check.get_experiment()
    ])
    batch.run()
    batch.plot()

if __name__ == '__main__':
    main()
