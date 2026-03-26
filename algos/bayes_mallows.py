from .algo import Algorithm
from feedback import PreferenceFeedback
import numpy as np
from typing import Any
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

# ---------------------------------------------------------
# Mute R Console Output (prevents tqdm progress bar spam)
# ---------------------------------------------------------
def ignore_r_output(x: str) -> None:
    pass

# Route both standard prints AND messages/warnings to the void
rpy2.rinterface_lib.callbacks.consolewrite_print = ignore_r_output
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = ignore_r_output

class BayesMallowsAlgorithm(Algorithm[PreferenceFeedback]):
    # When initializing the algorithm, you need much higher nmc for this to stabilize
    def __init__(self, n: int, nmc: int = 40000, burnin: int = 1000, metric: str = "footrule") -> None:
        super().__init__(name="BayesMallows")
        self.n = n
        self.nmc = nmc
        self.burnin = burnin
        self.metric = metric
        self.comparisons: list[dict[str, Any]] = []
        
        try:
            self.bm = importr("BayesMallows")
        except Exception as e:
            raise ImportError("The R package 'BayesMallows' is missing.") from e

    def feedback(self, feedback: PreferenceFeedback) -> None:
        self.comparisons.append({
            "assessor": 1, 
            "top_item": feedback.prec_id,
            "bottom_item": feedback.succ_id 
        })

    def predict(self) -> list[int]:
        if len(self.comparisons) == 0:
            return np.random.permutation(self.n).tolist()
        
        df = pd.DataFrame(self.comparisons)
        df["top_item"] = df["top_item"] + 1
        df["bottom_item"] = df["bottom_item"] + 1
        
        # Use the suppressor to keep the terminal clean
        with suppress_stdout_stderr():
            try:
                with localconverter(ro.default_converter + pandas2ri.converter):
                    r_df = ro.conversion.py2rpy(df)
                    
                data_obj = self.bm.setup_rank_data(preferences=r_df)
                model_options = self.bm.set_model_options(metric=self.metric, error_model="bernoulli")
                compute_options = self.bm.set_compute_options(nmc=self.nmc, burnin=self.burnin)
                
                model = self.bm.compute_mallows(
                    data=data_obj,
                    model_options=model_options,
                    compute_options=compute_options
                )
                
                consensus_df = self.bm.compute_consensus(model, type="CP")
                
                with localconverter(ro.default_converter + pandas2ri.converter):
                    consensus_pd = ro.conversion.rpy2py(consensus_df)
                    
            except Exception as e:
                # Note: This print might not show up if the suppressor is active, 
                # but we've exited the 'with' block by this point.
                return np.random.permutation(self.n).tolist()
            
        # 4. Extract items, sort, and shift back to 0-indexed for Python
        sorted_items = consensus_pd.sort_values(by="ranking")['item'].tolist()
        
        # Safely extract just the digits from R's auto-generated labels (e.g., "Item 16" -> 16)
        sorted_ids = [int(''.join(filter(str.isdigit, str(x)))) - 1 for x in sorted_items]
        
        # Catch-all: ensure any completely unranked items left out by R are appended
        missing_items = list(set(range(self.n)) - set(sorted_ids))
        if missing_items:
            sorted_ids.extend(missing_items)
            
        return sorted_ids
