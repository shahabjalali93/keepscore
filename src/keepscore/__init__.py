"""
keepscore — does your signal predict anything, or does it just sound right?

    from keepscore import score_claim

    rows = [{"value": True,  "outcome": "win",  "r": 2.1}, ...]
    print(score_claim(rows)["why"])

Full documentation in the README.
"""

from .core import (
    FINAL_OUTCOMES,
    Thresholds,
    score_claim,
    scorecard,
    selection_edge,
    weigh,
)

__version__ = "0.1.0"
__all__ = [
    "Thresholds", "score_claim", "scorecard", "weigh", "selection_edge",
    "FINAL_OUTCOMES", "__version__",
]
