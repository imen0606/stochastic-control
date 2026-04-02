# Financial RLVR Gym — Design Specification

**Date:** 2026-04-02
**Author:** Collaborative design session
**Status:** Approved for implementation planning

---

## Background

This gym implements the framework from Bilokon (2026), *"Strategy-Relative Market Regimes as
Filtration Compressions"*, as a verifiable reinforcement learning environment for RLVR
post-training of frontier language models.

The key insight: market regimes defined relative to a trading strategy have a computable optimal
policy (via Bellman backward induction), making them suitable for RLVR where rewards must be
exactly verifiable.

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

**Difficulty is controlled entirely through parameter ranges:**

| Parameter | Low difficulty | High difficulty |
|---|---|---|
| `lam` | 0.0 (greedy = optimal) | 0.30 (planning required) |
| `alpha` | 0.50 (strong signal) | 0.10 (weak signal) |
| `T` | 3 (short horizon) | 20 (long horizon) |

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
score = (J(s_model) − J(s_random)) / max(J(s_star_realized) − J(s_random), ε)
```

where `ε = 1e-6` and `s_random` uses `seed = problem.seed`.

- Score = 1.0: optimal performance (by construction, for the optimal agent)
- Score = 0.0: random-level performance
- Score < 0: worse than random
- Score > 1.0: not possible (model cannot exceed the realised optimal)

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
  Initial regime:   s₋₁ = 0

At each step you observe signal Z_t and decide to activate (s_t=1)
or deactivate (s_t=0) the strategy. Switching from your previous
decision costs λ = 0.15, deducted from that step's PnL.

You will receive one observation at a time. At each step, reason
about the immediate expected PnL, the switching cost, and whether
the signal is likely to persist before stating your decision.
```

**Each subsequent turn — one step at a time:**
```
User:       t=0 | Z_0 = +0.42 | Previous regime: s₋₁ = 0
Assistant:  [reasoning about E[X] = 0.30 × 0.42 = 0.126 vs switching cost 0.15]
            s_0 = 0

User:       t=1 | Z_1 = +0.61 | Previous regime: s_0 = 0
Assistant:  [reasoning] ... s_1 = 1

User:       t=2 | Z_2 = -0.12 | Previous regime: s_1 = 1
Assistant:  [reasoning] ... s_2 = 0
...
```

"Previous regime" in each user turn reflects the model's actual prior decision (not the optimal).
This requires `prompts.py` to be stateful.

### Stateful Conversation Manager

`prompts.py` exposes two functions:

```python
def setup_prompt(problem: RegimeSwitchingProblem) -> str:
    # Returns Turn 0 setup message

def step_prompt(t: int, z_t: float, prev_regime: int) -> str:
    # Returns the user message for turn t
    # prev_regime = model's actual decision at t-1 (tracked externally)
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

```
Easy:   T=3,  λ=0.05, α=0.40
Medium: T=10, λ=0.15, α=0.30
Hard:   T=20, λ=0.30, α=0.20
```

### Procedure

Run all three agents on N=500 instances per difficulty level (different seeds). Compute mean
regret-normalised score per agent per level.

### Pass Criteria

| Condition | What it proves |
|---|---|
| Optimal score = 1.0 exactly at all levels | Solver and verifier are consistent (mathematical invariant — any deviation is a bug) |
| Greedy score > 0.1 at easy level | Gym is not unsolvable |
| Greedy < Optimal at all levels | Benchmark is non-trivial |
| Greedy-optimal gap increases easy → hard | Planning difficulty scales with λ |
| Random score ∈ [−0.1, 0.1] at all levels | Normalization is well-calibrated |

The **greedy-optimal gap widening** from easy to hard is the primary evidence for lab reviewers:
it proves the gym specifically tests inter-temporal planning, not just signal classification.

### Expected Output

```
Goldilocks Validation Report
─────────────────────────────────────────
             Easy    Medium    Hard
Random       0.01    -0.01     0.02
Greedy       0.72     0.51     0.28
Optimal      1.00     1.00     1.00
─────────────────────────────────────────
Greedy-Optimal gap:  0.28  →  0.49  →  0.72   ✓ monotone increasing
All conditions: PASS
```

Numbers are illustrative. Optimal row is guaranteed by construction, not estimated.

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
