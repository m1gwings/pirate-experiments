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

# ---------------------------------------------------------
# Mute R Console Output (prevents tqdm progress bar spam)
# ---------------------------------------------------------
def ignore_r_output(x: str) -> None:
    pass

# Route both standard prints AND messages/warnings to the void
rpy2.rinterface_lib.callbacks.consolewrite_print = ignore_r_output
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = ignore_r_output

class BayesMallowsAlgorithm(Algorithm[PreferenceFeedback]):
    def __init__(self, n: int, nmc: int = 2000, burnin: int = 500, metric: str = "footrule") -> None:
        super().__init__(name="BayesMallows")
        self.n = n
        self.nmc = nmc
        self.burnin = burnin
        self.metric = metric
        
        # Store feedback as a list of dictionaries mapping to the R dataframe structure
        self.comparisons: list[dict[str, Any]] = []
        
        # Import the BayesMallows package from the R environment
        try:
            self.bm = importr("BayesMallows")
        except Exception as e:
            raise ImportError(
                "The R package 'BayesMallows' is missing or rpy2 is misconfigured. "
                "Please ensure your Conda environment is correctly set up."
            ) from e

    def feedback(self, feedback: PreferenceFeedback) -> None:
        # Following convention: prec_id is the winner, succ_id is the loser.
        # BayesMallows explicitly uses 'top_item' and 'bottom_item'.
        # We assign a unique 'assessor' to each pair to handle transitive closure.
        self.comparisons.append({
            "assessor": len(self.comparisons) + 1, 
            "top_item": feedback.prec_id,
            "bottom_item": feedback.succ_id
        })

    def predict(self) -> list[int]:
        if len(self.comparisons) == 0:
            return np.random.permutation(self.n).tolist()
        
        df = pd.DataFrame(self.comparisons)
        
        # Shift to 1-indexed for R's internal graph constructors
        df["top_item"] = df["top_item"] + 1
        df["bottom_item"] = df["bottom_item"] + 1
        
        try:
            # 1. Convert pandas to R dataframe (inside context)
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(df)
                
            # 2. Run the MCMC and Consensus OUTSIDE the context manager.
            # This ensures 'model' remains a pure R object, not a Python dict!
            model = self.bm.compute_mallows(
                preferences=r_df,
                metric=self.metric,
                nmc=self.nmc
            )
            
            consensus_df = self.bm.compute_consensus(model, type="CP", burnin=self.burnin)
            
            # 3. Convert R dataframe back to pandas (inside context)
            with localconverter(ro.default_converter + pandas2ri.converter):
                consensus_pd = ro.conversion.rpy2py(consensus_df)
                
        except Exception as e:
            print(f"\nBayesMallows MCMC sampling failed: {e}")
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
