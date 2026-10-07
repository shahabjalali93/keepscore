"""
The tests that matter here are the ones that try to make it say yes when it
should say no. A library whose whole value is refusing to overclaim has to be
tested on its refusals, not its agreements.

    python -m pytest -q        (or: python tests/test_core.py)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keepscore import (  # noqa: E402
    Thresholds, score_claim, scorecard, selection_edge, weigh,
)

LOOSE = Thresholds(resamples=400)          # faster; same behaviour


def rows(n, value, mean_r, spread=0.6, seed=1):
    """n observations on one side of a claim, R multiples around mean_r."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        r = rng.gauss(mean_r, spread)
        out.append({"value": value, "r": round(r, 3),
                    "outcome": "win" if r > 0 else "loss"})
    return out


# --- the refusals -----------------------------------------------------------

def test_refuses_under_the_sample_floor():
    s = score_claim(rows(10, True, 1.0) + rows(10, False, -1.0), LOOSE)
    assert not s["usable"]
    assert s["verdict"] == "unproven"
    assert "needs 30" in s["why"]


def test_a_claim_true_on_everything_predicts_nothing():
    """No false side means no comparison, however good the win rate looks."""
    s = score_claim(rows(200, True, 0.8), LOOSE)
    assert not s["usable"], "a claim with no counterexamples was called usable"


def test_pure_noise_is_not_significant():
    """Same distribution both sides: the interval must straddle zero."""
    both = rows(120, True, 0.2, seed=7) + rows(120, False, 0.2, seed=8)
    s = score_claim(both, LOOSE)
    assert not s["usable"]
    assert s["verdict"] in ("not significant", "no effect")


def test_real_but_tiny_difference_is_reported_as_too_small():
    big = Thresholds(min_per_side=30, min_edge=0.50, resamples=400)
    s = score_claim(rows(400, True, 0.30, 0.2, seed=3) +
                    rows(400, False, 0.10, 0.2, seed=4), big)
    assert not s["usable"]
    assert s["verdict"] == "no effect"


def test_unresolved_rows_are_ignored_not_counted_as_losses():
    data = rows(40, True, 0.5) + rows(40, False, 0.0)
    data += [{"value": True, "outcome": "open", "r": None} for _ in range(500)]
    s = score_claim(data, LOOSE)
    assert s["true"]["n"] == 40, "open positions leaked into the count"


# --- the things it should find ---------------------------------------------

def test_finds_a_real_warning():
    s = score_claim(rows(80, True, -0.5, seed=11) +
                    rows(120, False, 0.4, seed=12), LOOSE)
    assert s["usable"]
    assert s["verdict"] == "predicts worse"
    assert s["edge_r"] < 0
    lo, hi = s["ci"]
    assert hi < 0, "the interval should sit entirely below zero"


def test_finds_a_real_positive_signal():
    s = score_claim(rows(100, True, 0.6, seed=21) +
                    rows(100, False, 0.0, seed=22), LOOSE)
    assert s["usable"] and s["verdict"] == "predicts better" and s["edge_r"] > 0


def test_verdict_is_stable_across_runs():
    data = rows(60, True, 0.4, seed=31) + rows(60, False, 0.0, seed=32)
    a, b = score_claim(data, LOOSE), score_claim(data, LOOSE)
    assert a["ci"] == b["ci"], "the bootstrap must be seeded"


# --- the scorecard and the weighing ----------------------------------------

def _mixed_card():
    claim_rows = []
    for r in rows(80, True, -0.5, seed=41):
        claim_rows.append(dict(r, claim="risk.stop_exposed"))
    for r in rows(120, False, 0.4, seed=42):
        claim_rows.append(dict(r, claim="risk.stop_exposed"))
    for r in rows(100, True, 0.2, seed=43):
        claim_rows.append(dict(r, claim="technical.momentum"))
    for r in rows(100, False, 0.2, seed=44):
        claim_rows.append(dict(r, claim="technical.momentum"))
    return scorecard(claim_rows, LOOSE)


def test_scorecard_separates_the_real_from_the_decorative():
    card = _mixed_card()
    assert card["claims"]["risk.stop_exposed"]["usable"]
    assert not card["claims"]["technical.momentum"]["usable"]
    assert card["sources"]["risk"]["standing"] == "earning its place"
    assert card["sources"]["technical"]["proven"] == 0


def test_weigh_is_exactly_zero_before_anything_is_proven():
    empty = scorecard([], LOOSE)
    w = weigh({"risk.stop_exposed": True}, empty)
    assert w["expected_edge_r"] == 0.0 and not w["used"]


def test_weigh_follows_the_sign_of_the_claim():
    card = _mixed_card()
    on = weigh({"risk.stop_exposed": True}, card)["expected_edge_r"]
    off = weigh({"risk.stop_exposed": False}, card)["expected_edge_r"]
    assert on < 0 < off


def test_weigh_ignores_unproven_claims():
    card = _mixed_card()
    w = weigh({"risk.stop_exposed": True, "technical.momentum": True}, card)
    assert len(w["used"]) == 1 and w["ignored"] == 1


# --- selection edge ---------------------------------------------------------

def _sel(n_taken, r_taken, n_skipped, r_skipped, seed=51):
    out = []
    for r in rows(n_taken, True, r_taken, seed=seed):
        out.append({"taken": True, "outcome": r["outcome"], "r": r["r"]})
    for r in rows(n_skipped, False, r_skipped, seed=seed + 1):
        out.append({"taken": False, "outcome": r["outcome"], "r": r["r"]})
    return out


def test_selection_says_nothing_with_nothing():
    assert selection_edge([], LOOSE)["verdict"] == "unknown"


def test_selection_names_the_missing_half():
    out = selection_edge(_sel(0, 0, 60, 0.2), LOOSE)
    assert "none marked as taken" in out["say"]


def test_selection_finds_a_trader_who_adds():
    out = selection_edge(_sel(60, 0.7, 90, 0.0), LOOSE)
    assert out["verdict"] == "you add" and out["selection_edge_r"] > 0


def test_selection_finds_a_trader_who_costs():
    out = selection_edge(_sel(60, 0.0, 90, 0.7), LOOSE)
    assert out["verdict"] == "you cost"
    assert "taking more of what they already call" in out["say"]


def test_selection_will_not_call_a_coin_flip_a_skill():
    out = selection_edge(_sel(60, 0.3, 60, 0.3, seed=61), LOOSE)
    assert out["verdict"] in ("indistinguishable", "neutral")


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("  ok   %s" % name)
                passed += 1
            except AssertionError as e:
                print("  FAIL %s: %s" % (name, e))
                failed += 1
    print("\n%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
