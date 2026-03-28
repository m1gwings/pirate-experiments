from .algo import Algorithm
from feedback import PartialOrderFeedback
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import re

pandas2ri.activate()

class BayesMallowsAlgorithm(Algorithm[PartialOrderFeedback]):
    def __init__(self, n: int, n_iter: int = 2000, burnin: int = 500) -> None:
        super().__init__(name="BayesMallows")
        self.n = n
        self.n_iter = n_iter
        self.burnin = burnin
        
        self.preferences = []
        
        ro.r('suppressPackageStartupMessages(library(BayesMallows))')
        
        # We define two paths inside R: one with the warm start, one without.
        ro.r('''
            run_mallows_fast <- function(pref_df, n_iter, burnin_val, rho_init_vec) {
                
                rank_data <- setup_rank_data(preferences = pref_df)
                
                # If Python passes a populated vector, use the warm start
                if (length(rho_init_vec) > 0) {
                    fit <- compute_mallows(
                        data = rank_data,
                        model_options = set_model_options(error_model = "bernoulli"),
                        compute_options = set_compute_options(nmc = as.numeric(n_iter)),
                        initial_values = set_initial_values(rho_init = rho_init_vec)
                    )
                } else {
                    # If data is missing items, skip warm start to prevent crashing
                    fit <- compute_mallows(
                        data = rank_data,
                        model_options = set_model_options(error_model = "bernoulli"),
                        compute_options = set_compute_options(nmc = as.numeric(n_iter))
                    )
                }
                
                burnin(fit) <- as.numeric(burnin_val)
                cons <- compute_consensus(fit)
                
                cons_ordered <- cons[order(cons$ranking), ]
                return(as.character(cons_ordered$item))
            }
        ''')
        self._r_fast_predict = ro.globalenv['run_mallows_fast']

    def feedback(self, feedback: PartialOrderFeedback) -> None:
        for pref in feedback.rank_breaking():
            self.preferences.append({
                'assessor': 1,
                'top_item': pref.prec_id + 1,
                'bottom_item': pref.succ_id + 1
            })

    def predict(self) -> list[int]:
        if not self.preferences:
            return list(range(self.n))

        # 1. Check if the universe of items is fully observed
        observed_items = set()
        for p in self.preferences:
            observed_items.add(p['top_item'])
            observed_items.add(p['bottom_item'])

        if len(observed_items) == self.n:
            # All items observed: Build the warm start vector
            win_counts = {i: 0 for i in range(1, self.n + 1)}
            for p in self.preferences:
                win_counts[p['top_item']] += 1
                
            warm_start_items = sorted(win_counts.keys(), key=lambda x: win_counts[x], reverse=True)
            
            rho_init = [0] * self.n
            for rank_idx, item_id in enumerate(warm_start_items):
                rho_init[item_id - 1] = rank_idx + 1

            r_rho_init = ro.IntVector(rho_init)
        else:
            # Missing items: Pass an empty vector to bypass the R crash
            r_rho_init = ro.IntVector([])

        # 2. Run the prediction
        df = pd.DataFrame(self.preferences)
        ordered_items = self._r_fast_predict(df, self.n_iter, self.burnin, r_rho_init)
        
        # 3. Safely extract and format the ranking
        ranking = []
        for item_val in ordered_items:
            match = re.search(r'\d+', str(item_val))
            if match:
                ranking.append(int(match.group()) - 1)
            
        missing_items = set(range(self.n)) - set(ranking)
        ranking.extend(list(missing_items))
        
        return ranking