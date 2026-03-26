from experiment.experiment import ExperimentBatch
from configs import example

def main():
    batch = ExperimentBatch(name="Example Batch", experiments=[
        example.get_experiment()
    ])
    batch.run()
    batch.plot()

if __name__ == '__main__':
    main()
