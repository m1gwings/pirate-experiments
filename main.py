from experiment.experiment import ExperimentBatch
from configs import sanity_check, trick_pl, boxcar_filter_trick

def main():
    # TODO: Set all the seeds

    batch = ExperimentBatch(name="Experiment Batch 1", experiments=[
        # sanity_check.get_experiment(),
        # trick_pl.get_experiment(),
        boxcar_filter_trick.get_experiment()
    ])
    batch.run()
    batch.plot()

if __name__ == '__main__':
    main()
