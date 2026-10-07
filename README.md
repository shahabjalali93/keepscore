# keepscore

Score whether a trading signal actually predicts anything — and refuse to answer
until it can.

No dependencies. Python 3.8+. One file you can read in fifteen minutes.

---

## Why this exists

I run a scalping bot. For five days it produced zero setups and raised zero
errors, and I thought it was being patient. One rule compared a value against a
string the code never emits. Always false, silently.

That bug taught me the cheap lesson — check your enums. The expensive lesson came
after I fixed it, when the setups came back and I had to decide which of my
indicators were doing anything. My dashboard showed every indicator next to a win
rate. Every one of them looked fine. Win rate next to a signal that fires on 92%
of your trades is not evidence of anything; it is your overall win rate wearing a
costume.

`keepscore` is the part I pulled out. It takes a signal, looks at what happened
the times it fired and the times it didn't, and gives you one of four answers.
Three of them are "no".

---

## What it does

```python
from keepscore import score_claim

rows = [
    {"value": True,  "outcome": "win",  "r": 1.8},
    {"value": False, "outcome": "loss", "r": -1.0},
    # ... one row per resolved trade
]

s = score_claim(rows)
print(s["verdict"])   # predicts better | predicts worse | not significant
                      # | no effect | unproven
print(s["why"])       # +0.39R better when true [+0.16, +0.62], over 198 and 202
print(s["usable"])    # True only if it cleared all three gates
```

`r` is the result in R multiples — reward over the risk you took, so `-1.0` is a
full stop-out. Rows whose `outcome` is neither a win nor a loss are treated as
unresolved and skipped, not counted as losses.

### The three gates

A signal is `usable` only if it passes all three:

1. **Sample floor** — at least 30 resolved outcomes *on each side*. A signal that
   fires on nearly everything has no false side to compare against, so it fails
   here no matter how good its win rate looks.
2. **Significance** — a 95% bootstrap interval on the difference in mean R that
   does not contain zero.
3. **Size** — at least 0.10R of difference. Real but tiny is still not tradeable
   after costs.

All three are adjustable:

```python
from keepscore import Thresholds
score_claim(rows, Thresholds(min_per_side=50, min_edge=0.25, confidence=0.99))
```

### Why a bootstrap and not a t-test

R multiples are not normally distributed. They pile up at exactly -1.0, because
that is where your stop is, and fan out above it with no symmetric tail below. A
test that assumes normality will call differences real more often than it should,
which is the one failure mode that matters here. The bootstrap assumes nothing
about the shape. It is seeded, so the same data gives the same verdict every time.

---

## The whole scorecard

```python
from keepscore import scorecard, weigh

card = scorecard(claim_rows)   # rows also carry a "claim" key
for name, s in card["claims"].items():
    print(name, s["verdict"], s["why"])
```

Name claims `source.thing` and you also get each source's standing — useful if
several agents, models or indicator families are each filing claims and you want
to know which of them is earning its place rather than which is loudest.

`weigh()` then applies only the proven ones to a new setup:

```python
w = weigh({"rsi.oversold": True, "trend.above_ma": True}, card)
w["expected_edge_r"]   # +0.393
w["ignored"]           # 1 — trend.above_ma never earned a weight
```

Before anything is proven, `weigh()` returns exactly `0.0`. A fresh install
behaves as though the library were not installed, which is correct: a confident
number built from nothing is worse than no number.

---

## The question a journal cannot answer

Your journal contains the trades you took. Its win rate is the strategy and your
discipline fused into one number, and you can spend months improving the wrong
one.

Score the setups you skipped too, on paper, by the same rule:

```python
from keepscore import selection_edge

print(selection_edge(rows)["say"])
# -0.16R apart, interval [-0.38, +0.08] includes zero.
# On this much data your picks and your passes cannot be told apart.
```

Both sides have to be measured the same way. Comparing your real P&L against
paper results answers a different question, because real fills carry slippage and
your position sizing.

---

## Run the example

```
python examples/my_indicators_dont_work.py
```

It builds 400 synthetic trades in which exactly one of four indicators has any
predictive value, and shows what the other three look like on the way out:

```
     news.event_day     not significant  +0.08R apart, but the 95% interval [-0.18, +0.31] includes zero
USE  rsi.oversold       predicts better  +0.39R better when true [+0.16, +0.62], over 198 and 202
     trend.above_ma     not significant  -0.16R apart, but the 95% interval [-0.52, +0.19] includes zero
     volume.spike       not significant  -0.06R apart, but the 95% interval [-0.30, +0.17] includes zero

1 of 4 indicators have a measured edge.
```

Look at the interval on `trend.above_ma`. It is nearly twice as wide as the
others, because that indicator is true on 92% of trades and there is barely a
false side to compare against.

---

## Tests

```
python -m pytest -q      # or: python tests/test_core.py
```

The tests are weighted toward the refusals, because a library whose value is
declining to overclaim has to be tested on when it says no: a claim that is true
on everything, pure noise on both sides, a real but tiny difference, unresolved
rows trying to leak in as losses, and the seeded bootstrap returning the same
interval twice.

---

## What this is not

- Not a backtester. It scores outcomes you already have.
- Not a signal generator and not a position sizer. `weigh()` returns an expected
  edge in R, not a size.
- Not a substitute for out-of-sample data. If you tune a signal and then score it
  on the same trades you tuned it on, this will happily confirm your own work
  back to you. Nothing in here can detect that; only holding data back can.

---

## Install

```
pip install -e .
```

Or copy `src/keepscore/core.py` into your project. It imports nothing but the
standard library and that is deliberate.

---

MIT. Pulled out of JARVIS, my own trading system, where it decides which of the
agents' claims are allowed to affect anything.

— Wizafsky
