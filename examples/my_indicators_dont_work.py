"""
The honest version of a backtest report.

This builds a fake year of trades in which exactly ONE of four indicators has any
predictive value, then asks keepscore which. The point is what it says about the
other three, and how it says it.

    python examples/my_indicators_dont_work.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keepscore import scorecard, selection_edge, weigh  # noqa: E402

rng = random.Random(7)
TRADES = 400


def build():
    """
    400 trades. The only indicator that matters is `rsi.oversold`; everything else
    is decoration, and `trend.above_ma` is true on nearly everything, which is how
    an indicator looks useful without being useful.
    """
    claim_rows, journal = [], []
    for _ in range(TRADES):
        oversold = rng.random() < 0.45
        above_ma = rng.random() < 0.92          # almost always true
        volume_spike = rng.random() < 0.40      # pure noise
        news_day = rng.random() < 0.25          # pure noise

        edge = 0.45 if oversold else -0.05
        r = round(rng.gauss(edge, 1.1), 3)
        outcome = "win" if r > 0 else "loss"

        # the trader skips when it feels wrong, which correlates with nothing
        taken = rng.random() < 0.35

        for name, value in (("rsi.oversold", oversold),
                            ("trend.above_ma", above_ma),
                            ("volume.spike", volume_spike),
                            ("news.event_day", news_day)):
            claim_rows.append({"claim": name, "value": value,
                               "outcome": outcome, "r": r})
        journal.append({"taken": taken, "outcome": outcome, "r": r})
    return claim_rows, journal


def main():
    claim_rows, journal = build()
    card = scorecard(claim_rows)

    print("=" * 74)
    print("%d trades, 4 indicators\n" % TRADES)
    for name, s in card["claims"].items():
        mark = "USE " if s["usable"] else "    "
        print("%s %-18s %-16s %s" % (mark, name, s["verdict"], s["why"]))

    print("\n%d of %d indicators have a measured edge." % (card["proven"], card["total"]))
    print("\nLook at the interval on trend.above_ma. It is nearly twice as wide as")
    print("the others, because the indicator is true on 92% of trades and there is")
    print("barely a false side to compare against. A dashboard would have shown it")
    print("next to a healthy win rate and you would have believed it.")

    print("\n" + "=" * 74)
    print("A trade where RSI is oversold and the trend filter agrees:\n")
    w = weigh({"rsi.oversold": True, "trend.above_ma": True,
               "volume.spike": False, "news.event_day": False}, card)
    print("  expected edge: %+.3fR" % w["expected_edge_r"])
    print("  %s" % w["note"])
    for u in w["used"]:
        print("    %-18s %s  %+.3fR" % (u["claim"], u["value"], u["contribution_r"]))

    print("\n" + "=" * 74)
    print("And the question a journal cannot answer:\n")
    print("  " + selection_edge(journal)["say"])


if __name__ == "__main__":
    main()
