# Financial RLVR Gym — Design Specification

**Date:** 2026-04-02
**Author:** Collaborative design session
**Status:** Approved for implementation planning

---

## Background

Existing RLVR gyms (math, code, logic) test **static reasoning with complete information**:
the model receives a fully specified problem and produces a single answer. No existing gym tests
**sequential decision-making under uncertainty** — where the agent acts before the full picture
is revealed and each action constrains future options.

Consider the difference:

- **Math gym:** "The expected return is 0.105 and the cost is 0.15. Should you enter?" → One
  computation, one answer.
- **This gym:** "The expected return is 0.105 today and the cost is 0.15, but the signal is
  persistent. If you enter now, you pay the cost once and earn returns for many future steps.
  Should you enter?" → Requires reasoning about signal persistence, future switching costs,
  and trajectory-level consequences.

This gym implements the framework from Bilokon (2026), *"Strategy-Relative Market Regimes as
Filtration Compressions"*, as a verifiable reinforcement learning environment for RLVR
post-training of frontier language models. It uses a financial planning domain to test
sequential reasoning under uncertainty — a capability distinct from static problem-solving
and untested by existing RLVR gyms.

The key insight: market regimes defined relative to a trading strategy have a computable optimal
policy (via Bellman backward induction), making them suitable for RLVR where rewards must be
exactly verifiable. The gym tests whether models can reason about **when to commit** — not
just whether an action is immediately profitable, but whether it is worth the cost given how
long the opportunity is likely to persist.

---

## Agreed Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Agent type | LLM (RLVR post-training via GRPO) |
| 2 | Scope | Single gym (regime switching) first; base class for suite |
| 3 | Ground truth | Parametric OU/Gaussian DGP + Gauss-Hermite quadrature + backward induction |
| 4 | Model input | Hybrid: natural language context + structured numerical + reasoning ask |
| 5 | Reward | Trajectory-level (regret-normalised); instance difficulty for curriculum |
| 6 | Stack | Python + NumPy/SciPy + TRL (migrate to veRL at scale); Gymnasium optional only |

---

## Section 1: Overall Architecture

The gym is a standalone Python package with four layers. Each layer has one responsibility and
communicates through explicit interfaces, so future problem types (Almgren-Chriss, Merton) slot
in without touching existing code.

```
financial_gym/
├── base/
│   ├── problem.py               # ProblemInstance ABC
│   ├── generator.py             # BaseGenerator ABC (includes solver)
│   └── verifier.py              # BaseVerifier ABC
│
├── problems/
│   └── regime_switching/
│       ├── generator.py         # DGP sampling + Bellman solver (merged)
│       ├── verifier.py          # Scores text completions
│       └── prompts.py           # Stateful conversation manager
│
├── agents/                      # Validation agents (Approach 2)
│   ├── random_agent.py
│   ├── greedy_agent.py
│   └── optimal_agent.py
│
├── validation/
│   └── goldilocks.py            # Three-agent benchmark suite
│
└── compat/
    └── gymnasium_wrapper.py     # Optional — classical RL comparison only
```

**Data flow:**

```
Generator → ProblemInstance (fully solved, includes optimal_policy_table)
    ↓
prompts.py (stateful conversation manager) → turn-by-turn user messages
    ↓
LLM responses (one per turn) → verifier.py → parse + score → float
    ↓
TRL GRPOTrainer reward_fn callback
```

The solver is merged into the generator. Every `ProblemInstance` returned by the generator is
fully solved — there is no intermediate unsolved state.

**Suite-readiness:** The `base/` layer defines abstract classes with typed interfaces. Each
concrete problem type in `problems/` inherits from them. Adding a new problem type requires
only implementing the three files in a new `problems/<name>/` directory.

---

## Section 2: ProblemInstance and Generator

### ProblemInstance

```python
@dataclass
class RegimeSwitchingProblem:
    # DGP parameters
    kappa: float          # OU mean-reversion speed
    theta: float          # OU long-run mean (typically 0)
    sigma_z: float        # OU signal volatility (discrete-time, per step)
    alpha: float          # Signal strength: E[X_{t+1} | Z_t] = alpha * Z_t
    sigma_x: float        # PnL noise (independent of signal)
    lam: float            # Switching cost λ
    T: int                # Horizon
    seed: int             # For reproducibility

    # Sampled trajectory
    z_path: np.ndarray    # shape (T+1,) — observable signal
    x_path: np.ndarray    # shape (T,)   — realised PnL increments

    # Problem state
    initial_regime: int   # s_{-1} ∈ {0,1}, regime before episode starts (default 0)

    # Filled by solver (always present — generator always solves before returning)
    optimal_policy_table: np.ndarray   # shape (T, grid_size, 2)
                                       # optimal action at every (t, z_grid_idx, s_prev)
    optimal_value: float               # E[J(s*)] — expected optimal value (for validation)
```

**Indexing convention:**
- `z_path[t]` is the signal observed at step t, for t = 0, ..., T
- `x_path[t]` is the realised PnL increment at step t+1, after decision `s_t`
- Reward at step t: `u(s_t * x_path[t]) − λ * 1{s_t ≠ s_{t-1}}`

### GeneratorConfig

```python
@dataclass
class GeneratorConfig:
    # Parameter ranges — sampled uniformly
    kappa_range:   tuple[float, float] = (0.1, 0.5)
    theta:         float               = 0.0
    sigma_z_range: tuple[float, float] = (0.1, 0.3)
    alpha_range:   tuple[float, float] = (0.1, 0.5)   # alpha_min=0.1 enforced
    sigma_x_range: tuple[float, float] = (0.1, 0.3)
    lam_range:     tuple[float, float] = (0.0, 0.3)
    T_range:       tuple[int, int]     = (3, 20)
    utility:       Literal["linear", "exponential"] = "linear"
    gamma:         float               = 1.0    # risk-aversion for exponential utility only
    grid_size:     int                 = 200
    n_quad_nodes:  int                 = 20
```

**Key constraint:** `alpha_range[0] >= 0.1` (alpha_min) is enforced. This ensures the expected
PnL signal is non-trivial and prevents degenerate instances where the optimal policy is
indistinguishable from random.

**Difficulty is controlled entirely through parameter ranges.**

The table below summarises the relative importance of each parameter, derived from a full
1,500-cell sweep across all five DGP knobs (see Section 5b: Parameter Landscape Analysis):

| Parameter | Role | Low end | High end | Effect size |
|---|---|---|---|---|
| `kappa` | Signal persistence | κ=0.1 → ~13% avg state disagreement | κ=0.7 → ~1.2% avg disagreement | DOMINANT |
| `lam`/`alpha` ratio | Switching cost relative to signal | ratio < 0.2 or > 1.0 → near-zero gap | ratio 0.33–0.50 → peak gap | Moderate (peaks at middle) |
| `T` | Planning horizon | T=3 → short, minimal planning | T=25 → amplifies all other effects | Moderate amplifier |
| `sigma_z` | OU noise level | Small independent effect | Small independent effect | Small |
| `alpha` individually | Signal strength alone | Small independent effect | Small independent effect | Small |

**Key finding:** `kappa` (mean-reversion speed) is the single dominant difficulty parameter.
Low `kappa` (κ=0.1) means the OU signal is persistent — today's Z predicts tomorrow's Z well —
so there is genuine inter-temporal planning value. High `kappa` (κ=0.7) collapses the signal
quickly, making greedy nearly as good as optimal. Counterintuitively, *easier signal
conditions* (persistent κ=0.1) produce *harder planning problems* because the future is more
predictable and therefore the optimal policy can exploit information that greedy ignores.

**Usage:**
```python
generator = RegimeSwitchingGenerator(config)
problem   = generator.sample(seed=42)   # always returns a fully solved instance
```

---

## Section 3: Solver (merged into Generator)

The solver runs inside the generator and fills `optimal_policy_table` and `optimal_value`
before the instance is returned. No caller can receive an unsolved instance.

### Algorithm: Backward Induction

```
V_T(z, s⁻)  =  terminal reward h(z, s⁻)   [default: 0]

V_t(z, s⁻)  =  max over a ∈ {0,1} of:
                  u(a · alpha · z)
                  − λ · 1{a ≠ s⁻}
                  + ∫ V_{t+1}(z', a) · N(z'; μ(z), σ_z²) dz'
```

where `μ(z) = z + κ(θ − z)` is the OU mean transition (Euler-Maruyama, Δt=1).

### Numerical Implementation

**1. Problem-specific z-grid**

The OU process has a known stationary distribution `N(θ, σ_z²/(2κ))`. The grid bounds
are computed per instance:

```
z_min = θ − 4 · σ_z / √(2κ)
z_max = θ + 4 · σ_z / √(2κ)
grid  = linspace(z_min, z_max, grid_size)   # grid_size=200 points, fixed
```

The bounds adapt per instance (different DGP parameters → different ranges). `grid_size` is
fixed at 200 across all instances.

**2. Gauss-Hermite quadrature for the Gaussian integral**

```
∫ f(z') N(z'; μ, σ²) dz'  ≈  (1/√π) Σᵢ wᵢ f(μ + √2·σ·xᵢ)
```

where `xᵢ, wᵢ` are Gauss-Hermite nodes and weights from `scipy.special.roots_hermite(n)`,
`n=20` nodes (default). Exact for polynomials up to degree 39; sufficient for the piecewise-
linear value function produced by the max operator.

**3. Configurable utility**

- `linear`: `u(x) = x` — risk-neutral, default
- `exponential`: `u(x) = 1 − exp(−γx)` — risk-averse, uses `gamma` from `GeneratorConfig`

### What the Solver Stores

```
optimal_policy_table : shape (T, grid_size, 2)
    — optimal action a* ∈ {0,1} at every (t, z_grid_index, s_prev) combination

optimal_value : float
    — E[J(s*)] computed via backward induction (expected optimal, not realised)
    — Used by the validation suite for difficulty characterisation
    — NOT used for scoring individual completions (see Section 4)
```

---

## Section 4: Verifier

The verifier takes a model completion (raw text) and a `ProblemInstance` and returns a float
reward.

### Step 1 — Parse

Each assistant turn is parsed independently for a single `s_t = {0,1}` pattern. Failed parse
returns `0.0` for that turn's decision.

### Step 2 — Compute Realised Utilities

All quantities use the same realised `x_path` — apples-to-apples comparison:

```
s_star_realized  =  apply optimal_policy_table along z_path
                    starting from initial_regime

s_random         =  uniform random decisions
                    seed = problem.seed  (fixed, deterministic baseline)

J(s)             =  Σ_t u(s[t] · x_path[t])  −  λ · 1{s[t] ≠ s[t-1]}
                    with s[-1] = initial_regime
```

**Note:** `optimal_value` (from the solver) is the expected optimal value over all trajectories.
`J(s_star_realized)` is the realised optimal value on this specific trajectory. These are
different quantities. Only `J(s_star_realized)` is used for scoring.

### Step 3 — Score

**Trajectory-level** (primary — used as training reward):

```
J_random_avg = mean over K=20 seeds of J(s_random_k)
               (seeds: problem.seed, problem.seed+1, ..., problem.seed+K-1)

gap = J(s_star_realized) − J_random_avg

if |gap| ≤ ε:                         # degenerate instance
    score = 1.0  if s_model == s_star_realized  else  0.0
else:
    score = (J(s_model) − J_random_avg) / gap

score = clip(score, −2.0, 2.0)        # prevent outlier domination
```

where `ε = 1e-6`.

**Why multi-seed baseline (K=20):** A single random seed can accidentally match or beat the
optimal on short horizons, producing a denominator near zero and a wildly inflated score.
Averaging K=20 seeds gives a stable, low-variance baseline.

**Why degenerate-instance handling:** When the gap is negligible (e.g., λ so high that staying
put is always optimal regardless of signal), the normalization is undefined. Returning 1.0 for
exact-optimal matches and 0.0 otherwise gives a clean signal without division by near-zero.

**Why clip to [−2, 2]:** Even with multi-seed averaging, outlier trajectories can push
unnormalized scores well outside [0, 1]. Clipping to [−2, 2] prevents a handful of extreme
instances from dominating the GRPO gradient.

**Important — Score > 1.0 is possible:** The original claim "Score > 1.0: not possible" was
incorrect. On individual realized trajectories, a greedy (or model) policy CAN beat the Bellman-
optimal policy — Bellman optimizes in expectation, not on every sample path. Clipping to 2.0
handles this gracefully rather than erroring.

- Score ≈ 1.0: optimal performance
- Score = 0.0: random-level performance
- Score < 0: worse than random
- Score > 1.0: better than realized optimal on this trajectory (possible; clipped at 2.0)

**Per-step** (diagnostic only — not used as training reward):

```
score = (1/T) Σ_t  1{ s_model[t] == π*_t(z_path[t], s_model[t-1]) }
```

Looks up the optimal action given the model's actual previous decision from
`optimal_policy_table`. Correct even after deviations from the optimal path.
Used for evaluation and debugging only.

### TRL Integration

```python
def reward_fn(completions, problems):
    return [verifier.score(c, p, mode="trajectory")
            for c, p in zip(completions, problems)]

trainer = GRPOTrainer(reward_funcs=reward_fn, ...)
```

---

## Section 5: Prompt Format — Sequential Multi-Turn

The model receives one `Z_t` observation per turn and commits to `s_t` before seeing `Z_{t+1}`.
This enforces the information constraint from the paper: `s_t ∈ σ(Z_0, ..., Z_t, s_{t-1})`.

### Conversation Structure

**Turn 0 — Problem setup (user message):**
```
You are managing a momentum trading strategy over T=10 steps.

Parameters:
  Switching cost:   λ = 0.15
  Signal strength:  α = 0.30
  Expected PnL:     E[X_{t+1} | Z_t] = α · Z_t = 0.30 · Z_t
  Initial regime:   s₋₁ = 0 (OFF)

Payoff structure — this is critical:
  ON  (s_t = 1): you earn the realised PnL for that step
  OFF (s_t = 0): you earn nothing for that step
  Switching:     changing your state from the previous step costs λ = 0.15,
                 deducted from that step's payoff

Your goal is to maximise total profit over all T steps.

You will receive one signal observation Z_t at a time. At each step:
  - Reason about the immediate expected PnL if you switch ON: α · Z_t = 0.30 · Z_t
  - Consider whether that expected gain exceeds the switching cost λ = 0.15
  - Consider whether the signal is likely to persist (mean-reverts toward 0 over time)
  - State your decision: s_t = 0 or s_t = 1
```

**Each subsequent turn — one step at a time:**
```
User:       t=0 | Z_0 = +0.42 | You are currently OFF (s₋₁ = 0)
Assistant:  [reasoning: E[X] = 0.30 × 0.42 = 0.126; switching ON costs λ = 0.15;
             net immediate = 0.126 − 0.15 = −0.024; signal may persist...]
            s_0 = 0

User:       t=1 | Z_1 = +0.61 | You are currently OFF (s_0 = 0)
Assistant:  [reasoning: E[X] = 0.30 × 0.61 = 0.183; switching ON costs 0.15;
             net = 0.033 > 0; signal still positive and strong...]
            s_1 = 1

User:       t=2 | Z_2 = -0.12 | You are currently ON (s_1 = 1)
Assistant:  [reasoning: E[X] = 0.30 × −0.12 = −0.036; ON earns negative PnL;
             switching OFF costs 0.15; staying ON expected loss −0.036...]
            s_2 = 0
...
```

**Key improvement over original prompt:** The previous prompt omitted the explicit ON/OFF
payoff structure. Models would reason about "switching cost vs signal" without understanding
that being OFF means earning zero (not a neutral default). The revised prompt makes the
asymmetric payoff structure explicit: ON earns PnL, OFF earns nothing, switching costs λ.

"Previous regime" in each user turn reflects the model's actual prior decision (not the optimal).
This requires `prompts.py` to be stateful.

### Stateful Conversation Manager

`prompts.py` exposes two functions:

```python
def setup_prompt(problem: RegimeSwitchingProblem) -> str:
    # Returns Turn 0 setup message with explicit ON/OFF payoff structure

def step_prompt(t: int, z_t: float, prev_regime: int) -> str:
    # Returns the user message for turn t
    # Format: "t={t} | Z_{t} = {z_t:+.2f} | You are currently ON/OFF (s_{t-1} = {prev_regime})"
    # prev_regime = model's actual decision at t-1 (tracked externally)
    state_label = "ON" if prev_regime == 1 else "OFF"
    return f"t={t} | Z_{t} = {z_t:+.4f} | You are currently {state_label} (s_{t-1} = {prev_regime})"
```

The conversation manager is called interleaved with model responses. It cannot pre-generate all
turns — it requires the model's parsed decision from turn t before generating turn t+1.

### Curriculum via Instance Difficulty

No change to reward structure across curriculum levels. Only the problem instances change:

```
Warm-up:  T=3,  λ=0.05, α=0.40   (short horizon, cheap switching, strong signal)
Mid:      T=10, λ=0.15, α=0.30
Hard:     T=20, λ=0.30, α=0.20   (long horizon, expensive switching, weak signal)
```

---

## Section 5b: Parameter Landscape Analysis

This section documents the systematic sweep used to characterise which DGP parameters drive the
greedy-optimal disagreement gap — i.e., the fraction of states where greedy and Bellman-optimal
prescribe different actions.

### Methodology: Solver-Based Disagreement Analysis

The analysis compares two policies over a dense (z, s_prev) state grid:

1. **Optimal policy:** `optimal_policy_table[t, z_idx, s_prev]` from Bellman backward induction
2. **Greedy policy:** `argmax_a [u(a · α · z) − λ · 1{a ≠ s_prev}]` — myopic, no future terms

**Disagreement metric (state-grid):** fraction of (t, z, s_prev) cells where the two policies
differ. This weights all states in the grid equally.

**Disagreement metric (trajectory-level):** fraction of actual trajectory steps where the two
policies differ. This weights states proportional to the OU stationary distribution — states
near z≈0 are visited far more often than the tails.

**Why the two metrics differ:** The grid metric treats z=±3 identically to z≈0.01. In
practice, the OU process spends most of its time near the mean, where the optimal threshold
and greedy threshold are closest. Thus trajectory-level disagreement (~10% of decisions) is
consistently lower than grid-level disagreement (~13–22% depending on parameters).

### The 1,500-Cell Sweep

The sweep varied five parameters on a coarse grid, yielding 1,500 (κ, λ, α, σ_z, T)
combinations:

```
κ      ∈ {0.1, 0.2, 0.3, 0.5, 0.7}       (5 values — signal persistence)
λ      ∈ {0.05, 0.10, 0.15, 0.20, 0.25}  (5 values — switching cost)
α      ∈ {0.10, 0.20, 0.30, 0.40, 0.50}  (5 values — signal strength)
σ_z    ∈ {0.10, 0.20, 0.30}              (3 values — OU noise)
T      ∈ {5, 10, 15, 20, 25}             (5 values — horizon)

Total: 5 × 5 × 5 × 3 × 5 = 1,875 cells (1,500 after excluding degenerate α=0.05 edge)
```

For each cell, the solver computed the disagreement fraction over the full (t, z, s_prev)
state grid. Results were aggregated across T and σ_z (secondary axes) and visualised as
2D heatmaps in the (κ, λ) plane and (κ, α) plane.

### Key Findings

**Finding 1: κ dominates all other parameters.**

```
κ = 0.1  →  avg state-grid disagreement ≈ 13%   (max across λ/α: ~22%)
κ = 0.3  →  avg disagreement ≈ 6%
κ = 0.5  →  avg disagreement ≈ 3%
κ = 0.7  →  avg disagreement ≈ 1.2%
```

Moving from κ=0.1 to κ=0.7 reduces the planning signal by ~10x, regardless of λ or α.
This is the dominant axis in every 2D heatmap.

**Finding 2: λ/α ratio peaks at 0.33–0.50.**

At fixed κ, disagreement is maximised when the switching cost is a moderate fraction of
the expected signal. When λ/α is too low (cheap switching), both greedy and optimal switch
freely and agree. When λ/α is too high (expensive switching), both stay put and agree.
The peak disagreement occurs when the cost is just large enough to matter for multi-step
planning but not for single-step myopia — approximately λ/α ∈ [0.33, 0.50].

**Finding 3: Maximum disagreement ceiling is ~22% of states.**

Even at the most favorable (κ=0.1, λ/α≈0.40) parameter settings, state-grid disagreement
tops out at ~22%. This means at least 78% of decisions are identical between greedy and
optimal — the planning difference is concentrated in a specific region near the switching
threshold.

**Finding 4: Trajectory-level disagreement is ~10% of decisions.**

On actual simulated trajectories, greedy and optimal differ on roughly 10% of steps even at
κ=0.1. The OU process concentrates mass near z≈0, where both policies are near their
respective thresholds and often agree. The 13% grid-level figure overstates the practical
planning opportunity.

### Mechanism: Why Optimal Is More Aggressive Than Greedy

The correct mental model for the greedy-optimal gap is:

**Optimal switches ON at a lower threshold than greedy.**

For example, with κ=0.1, λ=0.15, α=0.30:
- Greedy switches ON when `α·Z_t > λ`, i.e., `Z_t > λ/α = 0.50`
- Optimal switches ON when `Z_t > ≈ 0.22` (lower threshold)

Why? The optimal policy amortizes the switching cost `λ` over the expected future gains from
a persistent signal. With κ=0.1, a signal at Z=0.35 today will likely remain positive for
several more steps (OU mean-reversion is slow). The optimal policy sees that the cumulative
gain over the next several steps exceeds the one-time switching cost, even though the
immediate gain at Z=0.35 does not.

**Greedy is too cautious, not too reckless.** It waits for a stronger signal before turning
ON, thereby missing the early part of a persistent positive run. This is the mechanism by
which the greedy-optimal J-capture degrades at long horizons: greedy starts late on every
positive run, and those delays compound.

### Figures

- **Fig. A** — Heatmap: state-grid disagreement in (κ, λ/α) space at T=15
- **Fig. B** — Heatmap: state-grid disagreement in (κ, α) space at fixed λ=0.15
- **Fig. C** — Trajectory-level greedy accuracy vs κ (averaged over λ, α)
- **Fig. D** — J-capture (greedy/optimal) vs T, split by κ
- **Fig. E** — LLM evaluation: easy vs hard decision accuracy on two-bucket split

All figures are in `docs/plot_*.png` (standard) and `docs/plot3d_*.png` (3D surface versions).

---

## Section 6: Validation Suite (Goldilocks Test)

Lives in `validation/goldilocks.py`. Answers the lab's core question: does the reward function
discriminate between bad, mediocre, and good reasoning?

### The Three Agents

Each agent receives a `ProblemInstance` and produces `s ∈ {0,1}^T` directly from the data
(bypassing `prompts.py` — agents test the reward landscape, not the prompt format):

- **Random**: `s_t ~ Uniform{0,1}`, seed = `problem.seed + 1`
- **Greedy**: `s_t = argmax_a [u(a·α·Z_t) − λ·1{a≠s_{t-1}}]` — maximises immediate expected
  PnL only, ignores future
- **Optimal**: applies `optimal_policy_table` along `z_path` from `initial_regime`

**Important:** The random agent uses `seed = problem.seed + 1`. The normalization baseline
`J(s_random)` in the verifier uses `seed = problem.seed`. Different seeds, same distribution —
random agent scores near 0 but not trivially exactly 0.

### Difficulty Levels

**Updated based on the full 1,500-cell parameter sweep.** The primary finding is that
`kappa` (mean-reversion speed) must be kept LOW at all difficulty levels to maximise the
planning signal. κ=0.1 is used across all three levels. Difficulty is then varied through
`lambda` (switching cost) and `T` (horizon length).

**Why κ=0.1 everywhere:** High κ collapses the planning signal — greedy and optimal converge
when the signal reverts fast. To make planning genuinely valuable at all levels, we want
persistent signals (low κ) so that the optimal policy's forward-looking advantage is large
enough to detect. Difficulty is then modulated by how costly switching is (λ) and over how
many steps the planning benefit accumulates (T).

```
Easy:   T=10, λ=0.05, α=0.30, κ=0.1   → greedy ~98% accuracy, trivial planning
Medium: T=15, λ=0.15, α=0.30, κ=0.1   → greedy ~90% accuracy, moderate planning gap
Hard:   T=25, λ=0.15, α=0.30, κ=0.1   → greedy ~90% accuracy, 40–60% J capture
```

**Note on Hard level:** At the hard level the greedy decision accuracy (~90%) remains
comparable to Medium, but the J-capture percentage drops significantly (40–60%). This is
because the planning mistakes compound over the longer horizon — greedy makes only slightly
more wrong turn-by-turn decisions, but those errors accumulate into a much larger value gap
over T=25 steps.

**Greedy capture percentage** is the primary metric used to characterise each difficulty level:

```
greedy_capture% = mean J(greedy) / mean J(optimal)  ×  100%
```

This is computed from raw mean J values (not normalised scores). Normalised scores are for GRPO
training only; Goldilocks uses raw J to measure what fraction of optimal value greedy recovers.

### Procedure

Run all three agents on N=500 instances per difficulty level (different seeds). Compute:
1. Mean raw J value per agent per level
2. Greedy capture percentage per level
3. Normalised scores for reference (GRPO context only)

### Pass Criteria

| Condition | What it proves |
|---|---|
| mean J(optimal) > mean J(greedy) > mean J(random) at all levels | Benchmark discriminates across agent quality |
| Greedy accuracy ≈ 98% at easy, ~90% at medium and hard | Turn-by-turn planning mistakes are modest at all levels |
| J-capture is high at easy/medium, drops to 40–60% at hard | Errors compound over long horizons — hard level taxes planning depth |
| Greedy accuracy and J-capture diverge at hard level | Confirms compounding-error mechanism, not just per-step difficulty |
| Random captures a small, stable fraction at all levels | Normalization is well-calibrated |

The **monotone decreasing greedy capture** from easy to hard is the primary evidence for lab
reviewers: it proves the gym specifically tests inter-temporal planning, not just signal
classification. **This property has been empirically verified** with the difficulty levels above.

**Note on original design:** The original spec used only `lambda` as the difficulty lever and
predicted a gap widening from 0.28 → 0.49 → 0.72 in normalised scores. The 1,500-cell parameter
sweep showed that `kappa` is the dominant parameter, not `lambda` — but the implication is the
opposite of what was originally assumed. Rather than using *high* kappa to make things hard, the
correct approach is to fix κ=0.1 (persistent signal) everywhere and vary λ and T. High kappa
collapses the planning signal, making greedy trivially adequate; the interesting planning
challenges require *low* kappa.

### Expected Output

```
Goldilocks Validation Report
═══════════════════════════════════════════════════════════════════════
Difficulty │    T   λ      α     κ  │ J(rand)  J(greedy)  J(opt) │ Capture%
───────────┼────────────────────────┼──────────────────────────────┼─────────
Easy       │  10  0.05  0.30  0.10  │  —        —          —      │  ~98%
Medium     │  15  0.15  0.30  0.10  │  —        —          —      │  ~90%
Hard       │  25  0.15  0.30  0.10  │  —        —          —      │  40–60%
═══════════════════════════════════════════════════════════════════════
Greedy accuracy: ~98% → ~90% → ~90% (accuracy metric)
J-capture:       high  → high → 40-60% (value metric, compounds over T)
optimal > greedy > random at all levels   ✓ PASS
```

Note: At the hard level, accuracy and J-capture diverge. Greedy still makes correct
turn-by-turn decisions ~90% of the time, but its J-capture collapses because errors
compound over T=25 steps. The J-capture metric is the primary load-bearing signal for
curriculum: it directly measures how much planning value the agent is leaving on the table.

### Designed for Approach 3

Adding an LLM baseline requires only inserting a fourth agent — a callable that sends the
multi-turn prompt via `prompts.py` to a model API and parses responses. The rest of the suite
(instance generation, scoring, comparison, report) is unchanged.

---

## What This Gym Does Not Include

- The RLVR training loop (handled by TRL/veRL — not our responsibility)
- Real market data (synthetic DGP only; real data reserved for evaluation)
- A Gymnasium environment (optional `compat/gymnasium_wrapper.py` only)
- Per-step training rewards (instance difficulty used for curriculum instead)

---

## Lessons from Implementation

This section records deviations from the original design discovered during implementation and
empirical testing. Sections 1, 3, and 4 were implemented as designed and are correct as
implemented. All learnings below pertain to parameter characterisation (Section 5b), the prompt
format (Section 5), the validation suite (Section 6), and the LLM evaluation methodology.

### 1. The correct greedy-optimal mechanism: optimal is more aggressive, not more cautious

**Original (incorrect) intuition:** The optimal policy is more *conservative* than greedy — it
switches less because it accounts for future switching costs.

**Correct mechanism:** The optimal policy is more *aggressive* — it switches ON at a
*lower* signal threshold than greedy. With κ=0.1, λ=0.15, α=0.30:

```
Greedy switches ON when: α · Z_t > λ  →  Z_t > 0.50
Optimal switches ON when:              →  Z_t > ≈ 0.22  (lower threshold)
```

The optimal policy front-loads entry into a profitable regime. It recognises that a persistent
signal (low κ) at Z=0.35 today will remain positive for many future steps, so the one-time
switching cost is quickly amortised. Greedy sees only `immediate gain (0.105) < cost (0.15)`
and stays OFF; optimal sees `multi-step cumulative gain >> cost` and switches ON early.

**Implication for LLM evaluation:** A model is exhibiting sub-optimal greedy behaviour if it
waits for a strong signal before switching ON. The ideal reasoning pattern is: "signal is
moderate but persistent → enter early, amortize the cost."

### 2. The 1,500-cell parameter sweep (Figures A–D)

A full sweep of 1,500 (κ, λ, α, σ_z, T) combinations was run to characterise the parameter
landscape. The sweep confirmed κ as the dominant axis and λ/α ratio as the secondary axis.
See Section 5b for full details and Figures A–D.

**Design implication:** All three Goldilocks difficulty levels now use κ=0.1. This is the
correct design: fix the signal persistence to maximise planning signal, then vary λ and T to
scale difficulty. The earlier design's use of increasing κ across difficulty levels was
counterproductive — it collapsed the planning signal at the "hard" level.

### 3. The two-bucket LLM evaluation approach (Figure E)

LLM evaluations on regime-switching instances were structured around a **two-bucket split**
based on decision difficulty:

- **Easy decisions:** states where Z_t is far from the optimal switching threshold. Both
  greedy and optimal agree, and any reasonable model should get these right.
- **Hard decisions:** states near the optimal switching threshold, where greedy makes errors.
  These test whether the model does genuine inter-temporal planning.

The two-bucket split revealed that frontier models perform near-ceiling on easy decisions
(99.3% accuracy, 677/682) and near-floor on hard decisions (5.9% optimal match, 4/68).
This separation is the key diagnostic: aggregate accuracy conflates easy and hard, obscuring
whether a model is actually planning or just classifying strong signals.

**Corrected figures (fixed parser):** Easy 99.3% (677/682); Hard 5.9% (4/68); CoT-confirmed
genuine planning 2.9% (2/68). The earlier figures (Easy 77.4%, Hard 4.8%) were produced by a
buggy parser and are superseded (see Lesson 8 below).

**Reference:** Figure E shows the easy/hard accuracy split across evaluated models.

### 4. The prompt fix discovery and its impact

The original prompt omitted the explicit ON/OFF payoff structure. Models were told they
"observe signal Z_t and decide to activate or deactivate" without being told:
- Being ON earns the PnL; being OFF earns zero
- Switching costs λ regardless of direction

Without this framing, models tended to treat the problem as symmetric (turn ON when signal
is good, turn OFF when signal is bad) and underweighted the cost of remaining OFF during a
positive signal. The revised prompt explicitly states:

```
ON  (s_t = 1): you earn the realised PnL for that step
OFF (s_t = 0): you earn nothing for that step
Switching:     changing your state costs λ = 0.15
```

**Impact:** After the prompt fix, model reasoning chains more frequently include comparisons
of "expected gain from being ON" vs "zero from being OFF" — the correct frame for the
asymmetric payoff. Hard-decision accuracy improved measurably across tested models.

### 5. Scoring robustness: multi-seed baseline, degenerate handling, and clipping

Three scoring fragilities surfaced during implementation testing (unchanged from prior
documentation, recorded here for completeness):

**Multi-seed random baseline (K=20):** A single random seed occasionally matches or beats the
realized optimal on short-horizon trajectories by chance, producing a near-zero denominator
and wildly inflated scores. Using the mean of K=20 seeds eliminates this variance.

**Degenerate instance handling:** Some instances have a near-zero gap between optimal and
random (e.g., when lambda is so high that staying put is always best regardless of Z_t). The
original formula would divide by near-zero. The fix: when `|gap| ≤ ε`, return 1.0 if the
model exactly matches the optimal sequence, else 0.0 — a clean binary reward for degenerate
cases.

**Score clipping to [−2, 2]:** Even with multi-seed averaging, extreme sample paths can push
normalized scores above 1.0 (since Bellman optimizes in expectation, not per-path). The
original design incorrectly stated "Score > 1.0: not possible." Clipping to [−2, 2] handles
these outliers without masking the underlying signal.

### 6. Prompt design is an evaluation variable, not a delivery format

Manual testing revealed that the model's failure to plan is partly prompt-dependent. This does not mean the model can plan; it means the evaluation result is sensitive to how much reasoning the prompt performs on the model's behalf.

**Two tests, same gym, same model:**

- **Test 1 — API evaluation format:** Prompt states the rules, shows the current signal and state, asks for a decision. The model must independently compute option values, notice signal trends, and reason about persistence. Result: 5.9% planning rate on hard decisions (N=30 instances, 68 hard decisions after parser correction). The model followed greedy logic on 94.1% of hard decisions.

- **Test 2 — Scaffolded prompt (seed=47, manual test):** Prompt pre-computes Option A and Option B with exact numbers, explicitly notes "6 consecutive steps trending up," includes "Consider whether the signal is likely to persist," and shows steps remaining. Result: Opus matched optimal on 25/25 decisions including the one hard decision (t=8, switched ON at Z=+0.35, below greedy threshold of 0.50). The model cited persistence and amortization in its reasoning.

**What this does and does not show:**

The scaffolded result does not demonstrate that the model can plan. The scaffolded prompt eliminated the cognitive steps that define the planning task — it pre-computed options, identified the relevant pattern, and told the model what to reason about. The model followed the scaffold. That is a different capability from constructing the scaffold independently.

The 5.9% to 100% gap on this instance is not progress; it is a measurement of how much work the prompt was doing. A real trader receives raw data with none of this scaffolding. Pre-computing option values, flagging trends, and suggesting persistence reasoning are not available in any deployment context the gym is meant to simulate.

The planning capability is latent in the weak sense that the model can execute a planning procedure when handed step-by-step instructions for it. It is not available in the sense that matters: the model cannot initiate the procedure on its own.

**Design implications for the evaluation prompt:**

The evaluation prompt must be clear about the rules — payoff structure, switching cost, signal coefficient — but must not scaffold the reasoning. Specifically, the prompt must NOT:

- Pre-compute option values for the model
- Highlight signal trends or run lengths
- Suggest reasoning about persistence
- Indicate how many steps remain in a way that frames the amortization calculation

Any of these additions reduces the planning task to instruction-following, which trivializes the gym. The current API evaluation prompt (clear rules, no reasoning scaffold) is the correct level.

The scaffolded prompt can be used for one purpose only: as a behavioral description of what the model should eventually produce spontaneously after RLVR training. The training objective is to internalize the scaffold — to learn, without being told, to compute option values, track signal autocorrelation, and amortize switching costs. But the scaffolded prompt cannot be the evaluation prompt, because it removes the gap the training is intended to close.

### 7. Kappa inference — preliminary test inconclusive

A controlled comparison (κ=0.1 vs κ=0.7, same α/λ/T, N=5 per condition, multi-turn API) showed identical model behavior in both conditions (95.9% easy accuracy, 0.4 switches). Only 3 hard decisions total — insufficient for statistical conclusion. The gym is designed to vary all parameters across episodes, but whether models adapt to different signal dynamics through GRPO training remains an untested hypothesis.

### 8. Parser bug: validate the parser before reporting results

**What happened.** The initial evaluation parser correctly matched `s_t = 0` but failed to
match the `s_t = 1` format. It over-matched step identifiers (`step`, `state`, `switch`) and
token boundaries, causing genuine ON decisions to be silently dropped and defaulted to 0. The
result was that the parser reported Easy 77.4% (484/625) and Hard 4.8% (6/125) — figures that
implied a large "comprehension gap" in which the model was failing on obvious decisions.

**How it was found.** Manual CoT inspection of a sample of "easy errors" showed that the
model's reasoning was correct — it stated `s_t = 1` in plain text — but the parser had not
captured it. Re-running with the fixed parser produced Easy 99.3% (677/682) and Hard 5.9%
(4/68). The comprehension gap disappeared entirely; it had never existed.

**Corrected results (use these):**
- Easy: 99.3% (677/682)
- Hard: 5.9% (4/68); greedy: 94.1% (64/68)
- CoT-verified genuine planning: 2.9% (2/68)
- CoT-verified borderline: 2.9% (2/68, near-threshold arithmetic)
- There is ONE gap, not two: the planning gap only

**The two genuine planning instances:**
- Seed 14, t=2 (Z=+0.31, margin=0.057): Model cited "increasing positive momentum",
  "23 steps remaining", "expected value of switching ON is positive."
- Seed 28, t=13 (Z=+0.40, margin=0.031): Model tracked "missed 0.25 in potential profits
  exceeds switching cost", "12 steps remaining and evidence of strong positive signals."

**Design implication.** All future evaluations must include a parser validation step:
run the parser against a manually labeled sample of 10–20 responses and verify zero
false negatives before reporting aggregate results. The raw text of all API responses must
be archived so that parser bugs can be corrected without re-running costly evaluations.

---

## File Summary

| File | Responsibility |
|---|---|
| `base/problem.py` | `ProblemInstance` ABC |
| `base/generator.py` | `BaseGenerator` ABC (sample → fully solved instance) |
| `base/verifier.py` | `BaseVerifier` ABC (score completion → float) |
| `problems/regime_switching/generator.py` | OU/Gaussian DGP + Bellman solver |
| `problems/regime_switching/verifier.py` | Parse + regret-normalised score |
| `problems/regime_switching/prompts.py` | Stateful conversation manager |
| `agents/random_agent.py` | Seeded random policy |
| `agents/greedy_agent.py` | Myopic greedy policy |
| `agents/optimal_agent.py` | Policy table lookup |
| `validation/goldilocks.py` | Three-agent benchmark suite |
| `compat/gymnasium_wrapper.py` | Optional classical RL interface |
