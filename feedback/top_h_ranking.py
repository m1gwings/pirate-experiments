# top_h_ranking.py
from .feedback import RankBreakableFeedback
from .preference import PreferenceFeedback
from item import Item

class TopHRankingFeedback(RankBreakableFeedback):
    def __init__(self, top_h: list[Item], others: list[Item]):
        # Extract IDs for quick validation
        top_h_ids = [item.id for item in top_h]
        others_ids = {item.id for item in others}

        # 1. Check for duplicates in top_h
        if len(top_h_ids) != len(set(top_h_ids)):
            raise ValueError("Invalid top_h: Contains duplicate items, which violates strict ordering.")

        # 2. Check for contradictions between top_h and others
        overlap = set(top_h_ids).intersection(others_ids)
        if overlap:
            raise ValueError(f"Contradiction: Items {overlap} cannot be in both top_h and others.")

        self.top_h = top_h
        self.h = len(top_h)
        self.others = others

    def rank_breaking(self) -> list[PreferenceFeedback]:
        prefs: list[PreferenceFeedback] = []

        for l in range(self.h):
            for m in range(l+1, self.h):
                prefs.append(PreferenceFeedback(self.top_h[l], self.top_h[m]))
            for i in self.others:
                prefs.append(PreferenceFeedback(self.top_h[l], i))
        
        return prefs
