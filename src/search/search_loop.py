from dataclasses import dataclass
from search.candidate import Candidate
from search.scorer import CandidateScorer


@dataclass
class SearchResult:
    best_candidate: Candidate
    best_score: float
    all_scores: list

class SearchLoop:
    def __init__(self, scorer: CandidateScorer):
        self._scorer = scorer

    def run(self, candidates: list) -> SearchResult:
        if not candidates:
            raise ValueError("candidates list is empty")
        scored = [(c, self._scorer.score(c)) for c in candidates]
        scored.sort(key=lambda x: x[1])
        best_candidate, best_score = scored[0]
        return SearchResult(
            best_candidate=best_candidate,
            best_score=best_score,
            all_scores=scored,
        )
