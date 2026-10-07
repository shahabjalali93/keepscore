"""
keepscore.core - does this signal predict anything, or does it just sound right?

The question people usually ask of a signal is "how do trades with it perform",
and that question cannot be answered. A signal that is true on every trade has a
win rate identical to the strategy's, and a dashboard will print it in large
friendly numbers.

The answerable question is comparative: do the trades where it was TRUE do better
than the trades where it was FALSE? That difference is the only thing a signal
can be said to be worth, and it is what this module measures.

Three refusals do most of the work:

  - under `min_per_side` outcomes on EACH side, there is no answer, only a number
  - a difference inside the noise is reported as no effect, not rounded up
  - a difference whose bootstrap interval straddles zero is not called real,
    however large it looks

Nothing here decides a trade. It decides which signals have earned the right to
be weighed, and it declines to answer until the data can.
"""

import random
import statistics

__all__ = [
    "Thresholds", "score_claim", "scorecard", "weigh", "selection_edge",
    "FINAL_OUTCOMES",
]

#: Outcomes that count as resolved. Anything else is still open and is ignored.
FINAL_OUTCOMES = ("win", "loss", "timeout")


class Thresholds:
    """
    How hard this is to convince. The defaults are deliberately annoying.

    min_per_side
        Resolved outcomes needed on EACH side before a verdict is given. 30 is a
        convention, not a law; raise it if a false positive is expensive.
    min_edge
        Smallest difference in mean R worth calling an effect. Below this, a real
        difference is still not a useful one.
    confidence
        Level for the bootstrap interval on the difference. The interval must
        exclude zero before a signal is called usable.
    resamples
        Bootstrap iterations. More is slower and barely different past ~2000.
    seed
        Fixed by default so the same data gives the same verdict every run. A
        verdict that changes between runs is not a verdict.
    """

    def __init__(self, min_per_side=30, min_edge=0.10, confidence=0.95,
                 resamples=2000, seed=12345):
        if min_per_side < 2:
            raise ValueError("min_per_side must be at least 2")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self.min_per_side = min_per_side
        self.min_edge = min_edge
        self.confidence = confidence
        self.resamples = resamples
        self.seed = seed

    def __repr__(self):
        return ("Thresholds(min_per_side=%d, min_edge=%g, confidence=%g)"
                % (self.min_per_side, self.min_edge, self.confidence))


DEFAULT = Thresholds()


# ---------------------------------------------------------------------------
# summarising one side
# ---------------------------------------------------------------------------

def _resolved(rows):
    return [r for r in rows if r.get("outcome") in FINAL_OUTCOMES]


def _rs(rows):
    """The R multiples of the resolved rows, 0.0 where R was not recorded."""
    return [float(r.get("r") or 0.0) for r in rows]


def _side(rows):
    dec = _resolved(rows)
    if not dec:
        return {"n": 0, "win_rate": None, "mean_r": None, "total_r": 0.0}
    wins = sum(1 for r in dec if r["outcome"] == "win")
    rs = _rs(dec)
    return {"n": len(dec),
            "win_rate": round(wins / len(dec) * 100, 1),
            "mean_r": round(statistics.fmean(rs), 4),
            "total_r": round(sum(rs), 4)}


def _bootstrap_ci(a, b, th):
    """
    Confidence interval for mean(a) - mean(b), by resampling both sides.

    A bootstrap rather than a t-test on purpose: R multiples are not normal (they
    pile up at -1 and fan out above it), and a test that assumes they are will
    call differences real more often than it should.
    """
    rng = random.Random(th.seed)
    n_a, n_b = len(a), len(b)
    diffs = []
    for _ in range(th.resamples):
        sa = statistics.fmean(rng.choices(a, k=n_a))
        sb = statistics.fmean(rng.choices(b, k=n_b))
        diffs.append(sa - sb)
    diffs.sort()
    tail = (1.0 - th.confidence) / 2.0
    lo = diffs[max(0, int(tail * len(diffs)) - 1)]
    hi = diffs[min(len(diffs) - 1, int((1.0 - tail) * len(diffs)))]
    return round(lo, 4), round(hi, 4)


# ---------------------------------------------------------------------------
# one signal
# ---------------------------------------------------------------------------

def score_claim(rows, thresholds=None):
    """
    Was this signal worth anything?

    rows
        An iterable of dicts, one per observation:
        ``{"value": True/False, "outcome": "win"|"loss"|"timeout", "r": float}``
        ``value`` is whether the signal fired. ``r`` is the result in R multiples
        (reward-to-risk units), so -1.0 is a full stop-out. Rows whose outcome is
        anything else are treated as unresolved and skipped.

    Returns a dict with ``true`` and ``false`` summaries, ``edge_r`` (the
    difference in mean R), ``ci`` (its bootstrap interval), a ``verdict`` and a
    plain-English ``why``. ``usable`` is True only when all three gates pass.
    """
    th = thresholds or DEFAULT
    rows = list(rows)
    t_rows = _resolved([r for r in rows if r.get("value")])
    f_rows = _resolved([r for r in rows if not r.get("value")])
    t, f = _side(t_rows), _side(f_rows)

    res = {"true": t, "false": f, "edge_r": None, "ci": None,
           "verdict": "unproven", "why": "", "usable": False}

    if t["n"] < th.min_per_side or f["n"] < th.min_per_side:
        res["why"] = ("needs %d resolved on each side (have %d true, %d false)"
                      % (th.min_per_side, t["n"], f["n"]))
        return res

    edge = round(t["mean_r"] - f["mean_r"], 4)
    lo, hi = _bootstrap_ci(_rs(t_rows), _rs(f_rows), th)
    res["edge_r"], res["ci"] = edge, (lo, hi)

    if lo <= 0.0 <= hi:
        res["verdict"] = "not significant"
        res["why"] = ("%+.2fR apart, but the %d%% interval [%+.2f, %+.2f] "
                      "includes zero — this could be luck"
                      % (edge, round(th.confidence * 100), lo, hi))
        return res

    if abs(edge) < th.min_edge:
        res["verdict"] = "no effect"
        res["why"] = ("%+.2fR apart over %d and %d — real but too small to act on"
                      % (edge, t["n"], f["n"]))
        return res

    res["usable"] = True
    if edge > 0:
        res["verdict"] = "predicts better"
        res["why"] = ("%+.2fR better when true [%+.2f, %+.2f], over %d and %d"
                      % (edge, lo, hi, t["n"], f["n"]))
    else:
        res["verdict"] = "predicts worse"
        res["why"] = ("%+.2fR when true [%+.2f, %+.2f] — a warning worth heeding, "
                      "over %d and %d" % (edge, lo, hi, t["n"], f["n"]))
    return res


# ---------------------------------------------------------------------------
# every signal, and who is carrying them
# ---------------------------------------------------------------------------

def scorecard(claim_rows, thresholds=None):
    """
    Score every signal at once.

    claim_rows
        ``{"claim": "risk.stop_exposed", "value": bool, "outcome": ..., "r": ...}``

    A claim named ``source.name`` is also grouped under ``source``, so if several
    agents, indicators or models are each filing claims you get their standing as
    well. A source's standing is how many of its claims earned their place, not
    how many it files.
    """
    th = thresholds or DEFAULT
    by_claim = {}
    for r in claim_rows:
        by_claim.setdefault(r["claim"], []).append(r)

    claims = {name: score_claim(rows, th) for name, rows in sorted(by_claim.items())}

    sources = {}
    for name, s in claims.items():
        src = name.split(".", 1)[0] if "." in name else "_"
        g = sources.setdefault(src, {"claims": 0, "proven": 0, "no_effect": 0,
                                     "unproven": 0, "best": None, "best_edge": 0.0})
        g["claims"] += 1
        if s["usable"]:
            g["proven"] += 1
            if abs(s["edge_r"]) > abs(g["best_edge"]):
                g["best_edge"], g["best"] = s["edge_r"], name
        elif s["verdict"] in ("no effect", "not significant"):
            g["no_effect"] += 1
        else:
            g["unproven"] += 1

    for g in sources.values():
        if g["proven"]:
            g["standing"] = "earning its place"
        elif g["no_effect"] == g["claims"]:
            g["standing"] = "measured, and says nothing useful"
        else:
            g["standing"] = "not enough resolved outcomes to judge"

    proven = sum(1 for s in claims.values() if s["usable"])
    return {"claims": claims, "sources": sources, "proven": proven,
            "total": len(claims), "thresholds": repr(th)}


# ---------------------------------------------------------------------------
# putting it to work
# ---------------------------------------------------------------------------

def weigh(claims, card):
    """
    What the signals, together, say about one new observation.

    claims
        ``{"claim_name": True/False}`` for the thing in front of you now.
    card
        The output of :func:`scorecard`.

    Only claims that earned their place contribute, and each contributes its own
    measured edge. Before anything is proven this returns exactly zero, so a fresh
    install behaves as though this library were not installed. That is correct: a
    confident number built from nothing is worse than no number.
    """
    used, total = [], 0.0
    scored = card.get("claims") or {}
    for name, value in (claims or {}).items():
        s = scored.get(name)
        if not s or not s.get("usable"):
            continue
        contrib = round(s["edge_r"] * (1 if value else -1), 4)
        total += contrib
        used.append({"claim": name, "value": bool(value),
                     "contribution_r": contrib, "why": s["why"]})
    used.sort(key=lambda u: -abs(u["contribution_r"]))
    return {
        "expected_edge_r": round(total, 4),
        "used": used,
        "ignored": len(claims or {}) - len(used),
        "note": ("nothing has earned a weight yet, so this is zero by design"
                 if not used else
                 "%d of %d signals have a measured edge" % (len(used), len(claims or {}))),
    }


# ---------------------------------------------------------------------------
# the selection question
# ---------------------------------------------------------------------------

def selection_edge(rows, thresholds=None):
    """
    Is the result the strategy, or is it you?

    rows
        Every resolved opportunity the strategy produced, taken or not:
        ``{"taken": bool, "outcome": ..., "r": ...}``

    A trading journal only contains the trades its owner took, so its win rate is
    the strategy and the choosing fused into one number with no way to separate
    them. You can spend months improving the wrong one.

    Score the skipped ones too — on paper, by the same rule — and the difference
    is what your judgement is worth. Both sides must be measured the same way:
    comparing real P&L against paper answers a different question, because real
    fills carry slippage and position size.
    """
    th = thresholds or DEFAULT
    rows = list(rows)
    taken_rows = _resolved([r for r in rows if r.get("taken")])
    skipped_rows = _resolved([r for r in rows if not r.get("taken")])
    taken, skipped, every = _side(taken_rows), _side(skipped_rows), _side(rows)

    out = {"strategy": every, "taken": taken, "skipped": skipped,
           "selection_edge_r": None, "ci": None, "verdict": "unknown", "say": ""}

    if every["n"] == 0:
        out["say"] = "Nothing has resolved yet."
        return out
    if taken["n"] == 0:
        out["say"] = ("%d opportunities resolved, none marked as taken. Until you "
                      "mark the ones you trade, this can measure the strategy but "
                      "not your choosing." % every["n"])
        return out
    if taken["n"] < th.min_per_side or skipped["n"] < th.min_per_side:
        out["say"] = ("needs %d resolved on each side (have %d taken, %d skipped) "
                      "before the difference means anything"
                      % (th.min_per_side, taken["n"], skipped["n"]))
        return out

    edge = round(taken["mean_r"] - skipped["mean_r"], 4)
    lo, hi = _bootstrap_ci(_rs(taken_rows), _rs(skipped_rows), th)
    out["selection_edge_r"], out["ci"] = edge, (lo, hi)

    if lo <= 0.0 <= hi:
        out["verdict"] = "indistinguishable"
        out["say"] = ("%+.2fR apart, interval [%+.2f, %+.2f] includes zero. On this "
                      "much data your picks and your passes cannot be told apart."
                      % (edge, lo, hi))
    elif abs(edge) < th.min_edge:
        out["verdict"] = "neutral"
        out["say"] = ("Your picks and your passes performed the same (%+.2fR apart). "
                      "The result is the strategy, not the choosing — work on the "
                      "rules." % edge)
    elif edge > 0:
        out["verdict"] = "you add"
        out["say"] = ("The ones you took beat the ones you skipped by %+.2fR "
                      "[%+.2f, %+.2f]. Something in your judgement is not written "
                      "down in the rules yet — find it and encode it."
                      % (edge, lo, hi))
    else:
        out["verdict"] = "you cost"
        out["say"] = ("The ones you SKIPPED did %+.2fR better [%+.2f, %+.2f]. The "
                      "strategy is ahead of your execution, so changing the rules "
                      "will not help — taking more of what they already call will."
                      % (abs(edge), lo, hi))
    return out
