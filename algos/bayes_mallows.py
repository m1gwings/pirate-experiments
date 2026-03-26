from .algo import Algorithm
from feedback import PreferenceFeedback
import numpy as np

# Requires: pip install rpy2 pandas
from typing import Any
import pandas as pd  # type: ignore
# Requires R installed with: install.packages("BayesMallows")
import rpy2.robjects as ro # type: ignore
from rpy2.robjects import pandas2ri #type: ignore
from rpy2.robjects.packages import importr #type: ignore

class BayesMallowsAlgorithm(Algorithm[PreferenceFeedback]):
    def __init__(self, n: int, nmc: int = 2000, burnin: int = 500, metric: str = "footrule") -> None:
        super().__init__(name="BayesMallows")
        self.n = n
        self.nmc = nmc
        self.burnin = burnin
        self.metric = metric
        
        # Store feedback as a list of dictionaries mapping to the R dataframe structure
        self.comparisons: list[dict[str, Any]] = []
        
        # Activate pandas to R dataframe conversion
        pandas2ri.activate()
        
        # Import the BayesMallows package from the R environment
        try:
            self.bm = importr("BayesMallows")
        except Exception as e:
            raise ImportError(
                "The R package 'BayesMallows' is missing or rpy2 is misconfigured. "
                "Please install it in your R environment: install.packages('BayesMallows')"
            ) from e

    def feedback(self, feedback: PreferenceFeedback) -> None:
        # Following convention: prec_id is the winner, succ_id is the loser.
        # BayesMallows explicitly uses 'top_item' and 'bottom_item'.
        # We assign a unique 'assessor' to each pair so that user-level cyclic 
        # inconsistencies do not break the strict transitive closure requirements.
        self.comparisons.append({
            "assessor": len(self.comparisons) + 1, 
            "top_item": feedback.prec_id,
            "bottom_item": feedback.succ_id
        })

    def predict(self) -> list[int]:
        if len(self.comparisons) == 0:
            return np.random.permutation(self.n).tolist()
        
        # 1. Convert comparisons to a pandas DataFrame, then to an R dataframe
        df = pd.DataFrame(self.comparisons)
        r_df = pandas2ri.py2rpy(df)
        
        try:
            # 2. Run the Bayesian MCMC algorithm
            # BayesMallows automatically handles the data augmentation for pairwise preferences
            # Note: Depending on your BayesMallows version, you might need to wrap r_df 
            # in setup_preferences(r_df) first.
            model = self.bm.compute_mallows(
                preferences=r_df,
                metric=self.metric,
                nmc=self.nmc,
                n_items=self.n
            )
            
            # 3. Extract the Cumulative Probability (CP) consensus ranking
            consensus_df = self.bm.compute_consensus(model, type="CP", burnin=self.burnin)
            consensus_pd = pandas2ri.rpy2py(consensus_df)
            
            # 4. Sort by the 'ranking' column and extract the corresponding items
            sorted_items = consensus_pd.sort_values(by="ranking")['item'].tolist()
            sorted_ids = [int(x) for x in sorted_items]
            
            # Catch-all: ensure any completely unranked items left out by R are appended
            missing_items = list(set(range(self.n)) - set(sorted_ids))
            if missing_items:
                sorted_ids.extend(missing_items)
                
            return sorted_ids
            
        except Exception as e:
            print(f"BayesMallows MCMC sampling failed: {e}")
            # Fallback to random if the chain fails (e.g., extremely sparse data)
            return np.random.permutation(self.n).tolist()
