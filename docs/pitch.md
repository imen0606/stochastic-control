# Financial RLVR Gym — Pitch

---

## The Problem

RLVR has transformed LLM post-training for math and code. But every existing
gym tests the same thing: **static reasoning with complete information**. The
model sees a full problem, produces one answer, gets scored.

No gym tests **sequential decision-making under uncertainty** — where the agent
acts before the full picture is revealed, each action has a cost, and the
optimal strategy requires reasoning about how conditions will evolve.

This capability — observe, infer, commit — is what agentic AI needs. And it
has no training signal today.

---

## Why Finance Has No RLVR Gym

RLVR requires a verifiable reward: a computable optimal to score against.

Financial markets have unknown, non-stationary distributions. You cannot
compute the optimal policy if you don't know the distribution. Existing
financial RL environments (FinRL, etc.) use realized PnL as reward — noisy,
path-dependent, unverifiable. Not suitable for RLVR.

```
┌─────────────────────────┐          ┌─────────────────────────┐
│    REAL MARKETS          │          │    MATH / CODE           │
│                          │          │                          │
│  Unknown distribution    │          │  Deterministic rules     │
│  → Can't compute optimal │          │  → Optimal is computable │
│  → Can't verify reward   │          │  → Reward is exact       │
│  → No RLVR possible     │          │  → RLVR works            │
└─────────────────────────┘          └─────────────────────────┘
            ↑                                    ↑
       BLOCKED                              SOLVED


                ┌─────────────────────────┐
                │    OUR GYM               │
                │                          │
                │  Synthetic signal (OU)   │
                │  Known distribution      │
                │  → Bellman gives exact   │
                │    optimal               │
                │  → Reward is verifiable  │
                │  → RLVR works            │
                │  + Financial domain      │
                │  + Sequential decisions  │
                │  + Uncertainty           │
                └─────────────────────────┘
                         ↑
                    THIS GYM
```

---

## How We Bypass It

We don't model the market. We model the **signal** that drives a trading
strategy.

Many financial signals — pairs trading spreads, momentum indicators,
volatility measures — exhibit **mean-reversion**. This is one of the most
documented properties in quantitative finance (Vasicek 1977, Uhlenbeck &
Ornstein 1930, empirical evidence in pairs trading: Gatev et al. 2006,
Liu et al. 2018).

We use the **Ornstein-Uhlenbeck process** for the signal:

```
Z_{t+1} = Z_t + κ(θ − Z_t) + σ_z · ε_t       ε_t ~ N(0,1)
```

This gives us three things no other financial process gives simultaneously:

1. **Financially meaningful dynamics.** Mean-reversion creates the commitment
   dilemma: should I act now on a weak signal that will persist, or wait for
   a stronger one that might revert?

2. **Gaussian transitions.** Z'|Z is Gaussian, so the Bellman continuation
   integral is computed exactly via Gauss-Hermite quadrature — not
   approximated. The optimal policy is provably correct.

3. **Bounded stationary distribution.** The signal stays in a predictable
   range (N(θ, σ²/2κ)), making discretisation reliable. Unlike GBM (variance
   grows unbounded) or jump-diffusion (non-Gaussian transitions), OU keeps
   computation exact and grids finite.

---

## The Framework

Built on Bilokon (2026), "Strategy-Relative Market Regimes as Filtration
Compressions" (SSRN 6504227).

The paper defines optimal binary regime processes — an agent observing signal
Z_t decides at each step whether to be ON (s_t=1, collect PnL) or OFF
(s_t=0, earn nothing), paying switching cost λ each time it changes state.
The objective:

```
J(s) = E[ Σ_t  u(s_t · X_{t+1}) − λ · 1{s_t ≠ s_{t-1}} ]
```

The paper proves existence and uniqueness of optimal regimes under Markov
structure. We make this computational: the Bellman recursion on a discretised
z-grid with Gauss-Hermite quadrature gives the exact optimal policy for any
parameter configuration.

---

## What Makes This Different

```
┌──────────────────┬──────────────┬────────────────┬──────────────────┐
│                  │ Math Gyms    │ FinRL          │ This Gym         │
├──────────────────┼──────────────┼────────────────┼──────────────────┤
│ Information      │ Complete     │ Partial        │ Partial          │
│ Decisions        │ Single       │ Sequential     │ Sequential       │
│ Uncertainty      │ None         │ Yes            │ Yes              │
│ Actions affect   │ No           │ Yes            │ Yes              │
│   future options │              │                │                  │
│ Verifiable       │ Yes          │ NO             │ Yes              │
│   reward         │              │                │                  │
│ Domain           │ Math/logic   │ Finance        │ Finance          │
│ Suitable for     │ Yes          │ No             │ Yes              │
│   RLVR           │              │                │                  │
└──────────────────┴──────────────┴────────────────┴──────────────────┘
```

We occupy the only cell that is: financial + sequential + uncertain +
verifiable.

---

## The Gym

```
Generator                    Verifier                  Lab's GRPO
   │                            │                         │
   ├─ Sample parameters         ├─ Parse LLM text         │
   │  (κ, α, λ, σ, T)          ├─ Compute model's J      │
   ├─ Simulate OU signal        ├─ Compute optimal's J    │
   ├─ Solve Bellman exactly     ├─ Regret-normalised      │
   ├─ Return fully-solved       │  score → float          │
   │  ProblemInstance            │                         │
   │                            │                         │
   └────── prompt ──────────────┴───── reward_fn ─────────┘
           (multi-turn)              (plugs into TRL)
```

Each episode has **different randomly drawn parameters**. The model doesn't
know κ (signal persistence) or σ_z (signal noise). It must infer these from
the signals it observes and adapt its strategy.

One function delivery: `reward_fn(completions, problems) → list[float]`

---

## What We Found

**Parameter landscape (1,500 configs, solver-based, exact):**
- Signal persistence (κ) is the dominant difficulty parameter
- At κ=0.1: ~10% of decisions require planning (greedy ≠ optimal)
- Those 10% account for 40-60% of the value gap
- Maximum planning demand ceiling: ~22% of states

**Evaluation of Claude Opus 4 (N=30 instances, 750 decisions):**
- Easy decisions (greedy = optimal): 77% accuracy
- Hard decisions (greedy ≠ optimal): 5% planning rate
- The model follows greedy logic on 95% of decisions where planning matters
- It adapts to known parameters (α, λ) but does not infer signal dynamics
- When scaffolded with explicit reasoning hints: plans correctly
- Without scaffolding: defaults to step-by-step cost-benefit (greedy)

**The mechanism:** The optimal policy is more aggressive than greedy — it
commits at weaker signals (threshold Z > 0.22 vs greedy's Z > 0.50) because
it amortises the one-time switching cost over persistent future gains. The
model cannot currently make this inference independently.

---

## What We Provide

- Open-source Python package (`pip install financial-rlvr-gym`)
- Bellman solver verified against analytical solutions
- 104 unit tests passing
- Multi-turn prompt format enforcing information constraint
- Three validation agents (random, greedy, optimal)
- Goldilocks benchmark suite
- Full parameter landscape analysis (1,500 configs)
- NeurIPS paper with all methodology and results
- Base class architecture for suite extension (Almgren-Chriss, Merton next)

---

## What We Don't Claim

- We don't claim models fail dramatically — they follow a reasonable heuristic
- We don't claim training on this gym improves general reasoning — untested
- We don't claim the OU process models real markets — it models the signal
- We don't claim 10% planning gap is large — but those decisions are expensive
- We have not run GRPO training — we provide the gym and baseline
