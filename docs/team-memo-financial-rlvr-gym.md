# Financial Planning RLVR Gym — Team Memo

**Date:** 2026-04-02
**Audience:** RLVR gym developers and stakeholders
**Status:** Design complete, implementation plan next

---

## 1. What We're Building

An RLVR gym where the model plays a portfolio manager deciding each day whether to run a
trading strategy or sit in cash.

Running the strategy makes or loses money depending on market conditions. Switching between
"in" and "out" has a cost — think brokerage fees and slippage. The model receives one market
signal per turn, reasons about it, and commits to a decision before seeing the next signal.

The optimal decision sequence is mathematically provable for every instance. That's our
verifiable reward.

---

## 2. Why This Gym Exists — The Gap in Current RLVR

Current RLVR gyms train two types of reasoning:

| Gym type | What it tests | Nature |
|---|---|---|
| Math (AIME, olympiad) | Multi-step deduction | Deterministic, single-shot |
| Code (contests, SWE-bench) | Logical execution | Deterministic, single-shot |

Both share the same structure: one problem, one answer, check if correct.

This gym tests something fundamentally different:

| | Math/Code Gyms | This Gym |
|---|---|---|
| Uncertainty | None — premises determine the answer | The model reasons about probabilities |
| Decisions | One | T coupled sequential decisions |
| Planning | Not needed | Each decision affects future options |
| Verification | Check final answer | Compare full trajectory against provable optimum |

**The capability gap:** No existing gym trains multi-step planning under uncertainty with
verifiable rewards. Agentic tasks (which labs care deeply about) require exactly this
capability — but there's been no good RLVR signal for it.

---

## 3. How It Works — Step by Step

### Step 1: Generate a Problem

We create a synthetic market scenario. The signal follows an Ornstein-Uhlenbeck process
(it drifts back toward zero over time — this captures mean-reversion, one of the most
well-documented properties of real financial signals).

**Example instance:**
```
Horizon:          T = 5 steps
Switching cost:   λ = 0.10 (paid each time you switch on↔off)
Signal strength:  α = 0.30
Signal path:      Z = [+0.60, +0.45, -0.20, -0.50, +0.35]
Realised PnL:     X = [+0.22, +0.10, -0.08, -0.18, +0.14]
```

**Why synthetic, not real data?** Because we defined the rules, we know the exact probability
distribution. If we used real market data, the distribution is unknown, the "optimal" policy
is an estimate, and the reward signal becomes noisy. RLVR breaks down.

The trade-off is explicit: realism for verifiability.

### Step 2: Solve for the Optimal Policy

We use **backward induction** (dynamic programming):

- Start at the last step. No future to worry about — just pick whichever action pays more.
- Step back one period. Your decision now = best immediate payoff + expected value of
  where that puts you next step.
- Repeat all the way to the beginning.

At each step, "expected value of next step" requires integrating over where the signal
might go next. Since we defined the signal process ourselves, this integral is exact
(computed via Gauss-Hermite quadrature — a numerical method exact for Gaussian distributions).

**The solver stores a full decision rule** — not just "what's optimal on this path" but
"what's optimal for any signal value, at any time step, from either previous state." This
is a lookup table of shape (T, 200 grid points, 2 previous states).

**Worked example using the instance above:**

```
Optimal decisions:   s* = [1, 1, 0, 0, 0]

t=0: Signal +0.60 → expected PnL = 0.18 > switching cost 0.10 → switch ON
t=1: Signal +0.45 → already on, no cost → stay ON
t=2: Signal -0.20 → negative, switch OFF (pay 0.10)
t=3: Signal -0.50 → already off → stay OFF
t=4: Signal +0.35 → PnL would be 0.14, but switching on costs 0.10,
     netting only 0.04 for one step → NOT WORTH IT → stay OFF
```

That last decision (t=4) is the interesting one — a greedy agent would switch on because
the immediate payoff is positive. The optimal agent knows it's the last step and the net
gain (0.04) is too small to justify the switch.

### Step 3: Present to the Model (Multi-Turn)

The model receives signals **one at a time** — it cannot see the future:

```
Setup:  "You manage a momentum strategy. Switching costs 0.10.
         Expected PnL = 0.30 × signal. One observation at a time."

Turn 1: "t=0 | Z = +0.60 | Previous: OFF"
Model:  "E[PnL] = 0.30 × 0.60 = 0.18. Exceeds switching cost 0.10.
         Signal is well above break-even. s_0 = 1"

Turn 2: "t=1 | Z = +0.45 | Previous: ON"
Model:  "Already on, no switching cost. E[PnL] = 0.135 > 0.
         Stay in. s_1 = 1"

Turn 3: "t=2 | Z = -0.12 | Previous: ON"
Model:  "Signal flipped negative. E[PnL] = -0.036. Staying on loses
         money. Switching off costs 0.10 but stops the bleeding.
         s_2 = 0"

... continues for all T steps
```

**Why one at a time?** If the model saw all signals upfront, it could use future information
to make perfect decisions that no real trader could make. Sequential revelation ensures the
model and the optimal benchmark play by the same rules.

### Step 4: Parse and Score

**Parse:** Extract `s_t = {0,1}` from each response. If a response is garbled, default to 0.

**Score:** Compute profit for three agents on the same market path:

```
Model's decisions:   [1, 1, 0, 0, 1]  →  earned $0.16
Random decisions:    [0, 1, 1, 0, 1]  →  lost  $0.14
Optimal decisions:   [1, 1, 0, 0, 0]  →  earned $0.12
```

Score formula:
```
score = (model − random) / (optimal − random)
      = (0.16 − (−0.14)) / (0.12 − (−0.14))
      = 0.30 / 0.26
      = 1.15
```

On individual instances, the model can score above 1.0 by getting lucky. Over hundreds of
instances, the optimal policy averages to 1.0 — it maximises expected value, not realised
value on every single path.

**Why not just count how many steps matched optimal?** Because not all steps matter equally.
Getting the decision right when the signal is strong and switching cost applies is worth much
more than getting it right when the signal is near zero. Profit-based scoring weights
decisions by their actual financial consequence.

### Step 5: Validate the Gym (Goldilocks Test)

Before shipping, we prove the reward function works using three synthetic agents — no LLM
needed:

- **Random:** coin flip decisions → ~0 score
- **Greedy:** react to current signal only, ignore future → middle score
- **Optimal:** Bellman solution → exactly 1.0

Run across three difficulty levels (500 instances each):

```
Goldilocks Validation Report (validated numbers)
──────────────────────────────────────────────────────────────────────
              Easy             Medium           Hard
              (κ=0.1, T=5)    (κ=0.3, T=10)   (κ=0.7, T=25)
Random        ~0               ~0               ~0
Greedy        0.985            0.849            0.613
Optimal       1.000            1.000            1.000
──────────────────────────────────────────────────────────────────────
Gap:          0.015            0.151            0.387
```

**The key number: the greedy-optimal gap.**

- Easy: greedy captures 98.5% of optimal (planning barely matters)
- Medium: greedy captures 84.9% (planning adds 15%)
- Hard: greedy captures 61.3% (planning adds 39%)

**The primary difficulty knob is kappa (κ), the signal reversion speed** — not just lambda.
The gap widens because at higher kappa the signal flips faster, creating more switching
traps for greedy agents. A greedy agent keeps switching in and out chasing each flip, paying
switching costs twice for nothing. You can't just react — you *must* think ahead about where
the signal is likely to be. This proves the gym specifically tests planning capability.

---

## 4. Lab Integration

The gym delivers one function:

```python
def reward_fn(completions, problems):
    return [verifier.score(c, p) for c, p in zip(completions, problems)]
```

Labs plug this into their existing TRL/veRL GRPOTrainer. No accommodation needed on their side.
Multi-turn GRPO is standard in all major training frameworks as of 2026.

---

## 5. Anticipated Questions

### "Why not use real market data?"
Real market distributions are unknown. Estimated "optimal" policies are noisy. The reward
signal becomes unreliable. By defining the market synthetically, we get exact ground truth.
We sacrifice realism for verifiability — same trade-off math gyms make (no one worries that
"solve 3x + 7 = 22" doesn't reflect real-world equations).

### "Why doesn't the model have tools (calculator, code interpreter)?"
If the model has a code interpreter, it can implement the Bellman solver directly and get
a perfect score every time. The gym becomes trivial — testing "can the model code a DP
algorithm" rather than "can the model reason about planning under uncertainty." That's what
code gyms already test. This gym tests internalised reasoning, same philosophy as math RLVR
(no calculator).

### "Is the Ornstein-Uhlenbeck process realistic?"
It captures mean-reversion, one of the most documented properties of financial signals.
It's simplified but not arbitrary. More importantly: the gym tests reasoning capability,
not market modelling. The OU process is the rules of the game. If the model learns to plan
correctly under known rules, that reasoning transfers to more complex settings.

### "One problem type isn't enough for meaningful post-training."
Correct — which is why the gym is designed as the first in a suite. The base class
architecture supports additional financial problems:

| Problem | Solution clarity | Delivery confidence |
|---|---|---|
| Regime switching (this gym) | Exact | Building now |
| Optimal execution (Almgren-Chriss) | Closed-form | Very high |
| Portfolio allocation (Merton) | Closed-form | Very high |
| Hedging with costs | Semi-analytical | Moderate-high |

Each problem type trains a different reasoning mode (switching costs, impact optimisation,
risk allocation) while sharing all infrastructure.

### "How does difficulty scale?"
Three knobs — no structural changes needed:
- **Switching cost (λ):** higher = more planning required
- **Signal strength (α):** weaker = noisier, harder to distinguish
- **Horizon (T):** longer = more decisions, harder credit assignment

### "Can the model score above 1.0?"
On individual instances, yes — by getting lucky on a specific market path. Over many
instances, the optimal policy averages to 1.0 because it maximises *expected* value.
An LLM that consistently scores above 1.0 would indicate a bug in the verifier.

### "Isn't greedy already near-optimal for this problem?"
With slow-moving signals (low kappa), yes — greedy captures 98% of optimal. The key
difficulty lever is kappa (mean-reversion speed). When signals revert fast (high kappa),
greedy switches too aggressively and pays switching costs twice for nothing. At hard
difficulty (κ=0.7, T=25), greedy captures only 61% of optimal — a 39% planning advantage.
This is tuneable: labs can adjust GeneratorConfig parameters without code changes.

---

## 6. Gaming Analogy

For teammates coming from game-based RLVR, here's how this maps:

| Game concept | Financial gym equivalent |
|---|---|
| Game state | Market signal Z_t (where is the signal now?) |
| Player action | Binary: activate (1) or sit out (0) |
| Action cost | Switching cost λ (paid every time you change your action) |
| Score/reward | Profit from your sequence of decisions |
| Optimal play | Bellman-optimal policy (computed exactly) |
| Difficulty setting | κ (signal reversion speed), λ (switching cost), T (horizon) |
| Fog of war | Signals revealed one at a time (can't see future) |
| Game engine | Ornstein-Uhlenbeck signal process (known rules) |

**The closest game analogy:** Imagine a turn-based game where each turn you decide to deploy
or recall a unit. Deploying earns points if conditions are favourable, loses if not.
Deploying/recalling costs resources. Conditions drift randomly but tend to revert to
neutral. The optimal strategy isn't "deploy when conditions are good" — it's "deploy when
conditions are good enough to justify the switching cost AND likely to stay good long enough."

That "long enough" reasoning is what makes this gym different from a simple classification
task. It requires forward-looking planning under uncertainty — the same capability needed
for any multi-step agentic decision-making.
