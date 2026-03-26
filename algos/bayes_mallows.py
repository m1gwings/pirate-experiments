from .algo import Algorithm
from feedback import PreferenceFeedback
import numpy as np
from typing import Any, Optional
import pandas as pd  # type: ignore

# rpy2 imports
import rpy2.robjects as ro # type: ignore
from rpy2.robjects import pandas2ri #type: ignore
from rpy2.robjects.packages import importr #type: ignore
from rpy2.robjects.conversion import localconverter #type: ignore
import rpy2.rinterface_lib.callbacks #type: ignore

import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to /dev/null."""
    with open(os.devnull, 'w') as fnull:
        old_stdout = os.dup(sys.stdout.fileno())
        old_stderr = os.dup(sys.stderr.fileno())
        try:
            os.dup2(fnull.fileno(), sys.stdout.fileno())
            os.dup2(fnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            os.dup2(old_stdout, sys.stdout.fileno())
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stdout)
            os.close(old_stderr)

def ignore_r_output(x: str) -> None:
    pass

rpy2.rinterface_lib.callbacks.consolewrite_print = ignore_r_output
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = ignore_r_output

class BayesMallowsAlgorithm(Algorithm[PreferenceFeedback]):
    def __init__(self, n: int, nmc: int = 10000, burnin: int = 1000, metric: str = "footrule", n_particles: int = 1000) -> None:
        super().__init__(name="BayesMallows")
        self.n = n
        self.nmc = nmc
        self.burnin = burnin
        self.metric = metric
        self.n_particles = n_particles
        
        # Buffer specifically for newly arriving feedback
        self.unprocessed_comparisons: list[dict[str, Any]] = []
        
        # Internal state tracking
        self._last_prediction: list[int] = np.random.permutation(self.n).tolist()
        
        # This will hold the persistent R model object across update steps
        self.mallows_model: Any = None
        
        try:
            self.bm = importr("BayesMallows")
        except Exception as e:
            raise ImportError("The R package 'BayesMallows' is missing.") from e

    def feedback(self, feedback: PreferenceFeedback) -> None:
        self.unprocessed_comparisons.append({
            "assessor": 1, 
            "top_item": feedback.prec_id,
            "bottom_item": feedback.succ_id 
        })

    def predict(self) -> list[int]:
        # Early exit if we have no new data to process
        if not self.unprocessed_comparisons:
            return self._last_prediction
        
        df = pd.DataFrame(self.unprocessed_comparisons)
        df["top_item"] = df["top_item"] + 1
        df["bottom_item"] = df["bottom_item"] + 1
        
        with suppress_stdout_stderr():
            try:
                with localconverter(ro.default_converter + pandas2ri.converter):
                    r_df = ro.conversion.py2rpy(df)
                    
                data_obj = self.bm.setup_rank_data(preferences=r_df)
                
                # Setup configuration options
                model_options = self.bm.set_model_options(metric=self.metric, error_model="bernoulli")
                
                if self.mallows_model is None:
                    # Step 1: Initialize the model via standard MCMC if this is the first batch of data
                    compute_options = self.bm.set_compute_options(nmc=self.nmc, burnin=self.burnin)
                    self.mallows_model = self.bm.compute_mallows(
                        data=data_obj,
                        model_options=model_options,
                        compute_options=compute_options
                    )
                else:
                    # Step 2: Use Sequential Monte Carlo (SMC) to update the existing posterior
                    smc_options = self.bm.set_smc_options(n_particles=self.n_particles)
                    self.mallows_model = self.bm.update_mallows(
                        model=self.mallows_model,
                        new_data=data_obj,
                        model_options=model_options,
                        smc_options=smc_options
                    )
                
                # Compute the new consensus ranking from the updated model
                consensus_df = self.bm.compute_consensus(self.mallows_model, type="CP")
                
                with localconverter(ro.default_converter + pandas2ri.converter):
                    consensus_pd = ro.conversion.rpy2py(consensus_df)
                    
            except Exception as e:
                # If the underlying R computation fails, fall back to the last known good ranking
                return self._last_prediction
            
        # Successfully processed this batch; clear the buffer
        self.unprocessed_comparisons.clear()
            
        # Extract items, sort, and shift back to 0-indexed for Python
        sorted_items = consensus_pd.sort_values(by="ranking")['item'].tolist()
        sorted_ids = [int(''.join(filter(str.isdigit, str(x)))) - 1 for x in sorted_items]
        
        # Catch-all: append any items completely unranked by the current data
        missing_items = list(set(range(self.n)) - set(sorted_ids))
        if missing_items:
            sorted_ids.extend(missing_items)
            
        # Update state with the newly found consensus
        self._last_prediction = sorted_ids
            
        return sorted_ids
