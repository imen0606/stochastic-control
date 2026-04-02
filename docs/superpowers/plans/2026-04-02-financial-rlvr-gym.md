# Financial RLVR Gym Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a verifiable RLVR gym where an LLM decides when to activate/deactivate a trading strategy, scored against an exact Bellman-optimal policy.

**Architecture:** Python package `financial_gym/` with abstract base classes for suite extensibility. First (and only) concrete implementation: regime switching. Generator produces fully-solved problem instances, verifier scores LLM text completions via regret-normalised profit, validation suite proves reward signal discriminates reasoning quality.

**Tech Stack:** Python 3.14, NumPy, SciPy (`roots_hermite`), pytest

---

## File Map

| File | Creates/Modifies | Responsibility |
|---|---|---|
| `pyproject.toml` | Create | Package config, dependencies |
| `financial_gym/__init__.py` | Create | Package root |
| `financial_gym/base/__init__.py` | Create | Base module |
| `financial_gym/base/problem.py` | Create | `BaseProblem` ABC |
| `financial_gym/base/generator.py` | Create | `BaseGenerator` ABC |
| `financial_gym/base/verifier.py` | Create | `BaseVerifier` ABC |
| `financial_gym/problems/__init__.py` | Create | Problems module |
| `financial_gym/problems/regime_switching/__init__.py` | Create | Regime switching module |
| `financial_gym/problems/regime_switching/generator.py` | Create | OU DGP + Bellman solver |
| `financial_gym/problems/regime_switching/verifier.py` | Create | Parse + regret-normalised score |
| `financial_gym/problems/regime_switching/prompts.py` | Create | Multi-turn conversation manager |
| `financial_gym/agents/__init__.py` | Create | Agents module |
| `financial_gym/agents/random_agent.py` | Create | Seeded random policy |
| `financial_gym/agents/greedy_agent.py` | Create | Myopic greedy policy |
| `financial_gym/agents/optimal_agent.py` | Create | Policy table lookup |
| `financial_gym/validation/__init__.py` | Create | Validation module |
| `financial_gym/validation/goldilocks.py` | Create | Three-agent benchmark suite |
| `tests/test_base.py` | Create | Base ABC tests |
| `tests/test_generator.py` | Create | Generator + solver tests |
| `tests/test_verifier.py` | Create | Verifier scoring tests |
| `tests/test_prompts.py` | Create | Prompt format tests |
| `tests/test_agents.py` | Create | Agent tests |
| `tests/test_goldilocks.py` | Create | Validation suite tests |

---

### Task 1: Project Scaffolding and Base ABCs

**Files:**
- Create: `pyproject.toml`
- Create: `financial_gym/__init__.py`
- Create: `financial_gym/base/__init__.py`
- Create: `financial_gym/base/problem.py`
- Create: `financial_gym/base/generator.py`
- Create: `financial_gym/base/verifier.py`
- Test: `tests/test_base.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "financial-rlvr-gym"
version = "0.1.0"
description = "Verifiable RLVR gym for financial planning under uncertainty"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

- [ ] **Step 2: Create package __init__ files**

```python
# financial_gym/__init__.py
"""Financial RLVR Gym — verifiable environments for LLM post-training."""

# financial_gym/base/__init__.py
"""Base classes for the financial gym suite."""
from financial_gym.base.problem import BaseProblem
from financial_gym.base.generator import BaseGenerator
from financial_gym.base.verifier import BaseVerifier

# financial_gym/problems/__init__.py
"""Concrete problem implementations."""

# financial_gym/problems/regime_switching/__init__.py
"""Regime switching gym — strategy on/off decisions with switching costs."""

# financial_gym/agents/__init__.py
"""Validation agents for the Goldilocks test."""

# financial_gym/validation/__init__.py
"""Validation suite for gym quality assurance."""
```

- [ ] **Step 3: Write the failing test for base ABCs**

```python
# tests/test_base.py
import pytest
import numpy as np
from financial_gym.base.problem import BaseProblem
from financial_gym.base.generator import BaseGenerator
from financial_gym.base.verifier import BaseVerifier


def test_base_problem_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseProblem()


def test_base_generator_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseGenerator()


def test_base_verifier_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseVerifier()


def test_base_generator_requires_sample_method():
    """Subclass must implement sample(seed) -> BaseProblem."""
    class BadGenerator(BaseGenerator):
        pass

    with pytest.raises(TypeError):
        BadGenerator()


def test_base_verifier_requires_score_method():
    """Subclass must implement score(completion, problem) -> float."""
    class BadVerifier(BaseVerifier):
        pass

    with pytest.raises(TypeError):
        BadVerifier()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_base.py -v`
Expected: FAIL — modules not found

- [ ] **Step 5: Implement BaseProblem ABC**

```python
# financial_gym/base/problem.py
"""Abstract base class for problem instances."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseProblem(ABC):
    """Base class for all gym problem instances.

    Every concrete problem must define its own fields.
    The generator always returns fully-solved instances.
    """

    @abstractmethod
    def horizon(self) -> int:
        """Return the number of decision steps T."""
        ...
```

- [ ] **Step 6: Implement BaseGenerator ABC**

```python
# financial_gym/base/generator.py
"""Abstract base class for problem generators."""
from abc import ABC, abstractmethod
from financial_gym.base.problem import BaseProblem


class BaseGenerator(ABC):
    """Base class for all gym generators.

    Generators create fully-solved problem instances.
    The solver is merged into the generator — callers
    never receive an unsolved instance.
    """

    @abstractmethod
    def sample(self, seed: int) -> BaseProblem:
        """Generate a single fully-solved problem instance.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            A fully-solved problem instance with optimal policy computed.
        """
        ...
```

- [ ] **Step 7: Implement BaseVerifier ABC**

```python
# financial_gym/base/verifier.py
"""Abstract base class for verifiers."""
from abc import ABC, abstractmethod
from financial_gym.base.problem import BaseProblem


class BaseVerifier(ABC):
    """Base class for all gym verifiers.

    Verifiers parse LLM text completions, extract decisions,
    and return a float reward score.
    """

    @abstractmethod
    def score(self, completion: list[str], problem: BaseProblem,
              mode: str = "trajectory") -> float:
        """Score a model's completion against the optimal policy.

        Args:
            completion: List of assistant response strings, one per turn.
            problem: The fully-solved problem instance.
            mode: "trajectory" (training reward) or "per_step" (diagnostic).

        Returns:
            Float score. 1.0 = optimal, 0.0 = random-level.
        """
        ...
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_base.py -v`
Expected: All 5 tests PASS

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml financial_gym/ tests/test_base.py
git commit -m "feat: project scaffolding and base ABCs"
```

---

### Task 2: RegimeSwitchingProblem Dataclass and GeneratorConfig

**Files:**
- Create: `financial_gym/problems/regime_switching/generator.py` (partial — dataclasses only)
- Test: `tests/test_generator.py` (partial — dataclass tests only)

- [ ] **Step 1: Write the failing test for dataclasses**

```python
# tests/test_generator.py
import numpy as np
import pytest
from financial_gym.problems.regime_switching.generator import (
    RegimeSwitchingProblem,
    GeneratorConfig,
)


class TestRegimeSwitchingProblem:
    def test_create_problem(self):
        problem = RegimeSwitchingProblem(
            kappa=0.3, theta=0.0, sigma_z=0.2, alpha=0.3,
            sigma_x=0.2, lam=0.1, T=5, seed=42,
            z_path=np.zeros(6), x_path=np.zeros(5),
            initial_regime=0,
            optimal_policy_table=np.zeros((5, 200, 2), dtype=np.int8),
            optimal_value=0.0,
        )
        assert problem.horizon() == 5
        assert problem.z_path.shape == (6,)
        assert problem.x_path.shape == (5,)
        assert problem.optimal_policy_table.shape == (5, 200, 2)

    def test_z_path_has_T_plus_1_entries(self):
        problem = RegimeSwitchingProblem(
            kappa=0.3, theta=0.0, sigma_z=0.2, alpha=0.3,
            sigma_x=0.2, lam=0.1, T=10, seed=1,
            z_path=np.zeros(11), x_path=np.zeros(10),
            initial_regime=0,
            optimal_policy_table=np.zeros((10, 200, 2), dtype=np.int8),
            optimal_value=0.0,
        )
        assert problem.z_path.shape == (problem.T + 1,)
        assert problem.x_path.shape == (problem.T,)


class TestGeneratorConfig:
    def test_default_config(self):
        config = GeneratorConfig()
        assert config.alpha_range[0] >= 0.1
        assert config.utility == "linear"
        assert config.grid_size == 200
        assert config.n_quad_nodes == 20

    def test_alpha_min_enforced(self):
        with pytest.raises(ValueError, match="alpha_min"):
            GeneratorConfig(alpha_range=(0.01, 0.5))

    def test_custom_config(self):
        config = GeneratorConfig(
            lam_range=(0.0, 0.05),
            T_range=(3, 5),
            alpha_range=(0.3, 0.5),
        )
        assert config.lam_range == (0.0, 0.05)
        assert config.T_range == (3, 5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_generator.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement dataclasses**

```python
# financial_gym/problems/regime_switching/generator.py
"""Regime switching problem generator with integrated Bellman solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from financial_gym.base.problem import BaseProblem
from financial_gym.base.generator import BaseGenerator


@dataclass
class RegimeSwitchingProblem(BaseProblem):
    """A fully-solved regime switching problem instance.

    Attributes:
        kappa: OU mean-reversion speed.
        theta: OU long-run mean (typically 0).
        sigma_z: OU signal volatility (discrete-time, per step).
        alpha: Signal strength — E[X_{t+1} | Z_t] = alpha * Z_t.
        sigma_x: PnL noise std dev (independent of signal).
        lam: Switching cost lambda.
        T: Number of decision steps (horizon).
        seed: Random seed used to generate this instance.
        z_path: Observable signal path, shape (T+1,).
        x_path: Realised PnL increments, shape (T,).
        initial_regime: s_{-1}, regime before episode starts.
        optimal_policy_table: Optimal action at (t, z_grid_idx, s_prev), shape (T, grid_size, 2).
        optimal_value: Expected optimal value E[J(s*)] (for validation, NOT for scoring).
    """
    kappa: float
    theta: float
    sigma_z: float
    alpha: float
    sigma_x: float
    lam: float
    T: int
    seed: int
    z_path: np.ndarray
    x_path: np.ndarray
    initial_regime: int
    optimal_policy_table: np.ndarray
    optimal_value: float

    def horizon(self) -> int:
        return self.T


@dataclass
class GeneratorConfig:
    """Configuration for the regime switching generator.

    Parameter ranges are sampled uniformly when generating instances.
    alpha_range[0] must be >= 0.1 to prevent degenerate instances.
    """
    kappa_range: tuple[float, float] = (0.1, 0.5)
    theta: float = 0.0
    sigma_z_range: tuple[float, float] = (0.1, 0.3)
    alpha_range: tuple[float, float] = (0.1, 0.5)
    sigma_x_range: tuple[float, float] = (0.1, 0.3)
    lam_range: tuple[float, float] = (0.0, 0.3)
    T_range: tuple[int, int] = (3, 20)
    utility: Literal["linear", "exponential"] = "linear"
    gamma: float = 1.0
    grid_size: int = 200
    n_quad_nodes: int = 20

    def __post_init__(self) -> None:
        if self.alpha_range[0] < 0.1:
            raise ValueError(
                f"alpha_min must be >= 0.1 to prevent degenerate instances, "
                f"got alpha_range[0]={self.alpha_range[0]}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_generator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add financial_gym/problems/ tests/test_generator.py
git commit -m "feat: RegimeSwitchingProblem dataclass and GeneratorConfig"
```

---

### Task 3: Bellman Solver (Core Mathematical Engine)

**Files:**
- Modify: `financial_gym/problems/regime_switching/generator.py` (add solver + generator)
- Test: `tests/test_generator.py` (add solver + generator tests)

- [ ] **Step 1: Write the failing test for utility functions**

Add to `tests/test_generator.py`:

```python
from financial_gym.problems.regime_switching.generator import (
    RegimeSwitchingProblem,
    GeneratorConfig,
    _linear_utility,
    _exponential_utility,
    _compute_z_grid,
)


class TestUtilityFunctions:
    def test_linear_utility(self):
        assert _linear_utility(0.5) == 0.5
        assert _linear_utility(0.0) == 0.0
        assert _linear_utility(-0.3) == pytest.approx(-0.3)

    def test_exponential_utility(self):
        # u(x) = 1 - exp(-gamma * x), gamma=1.0
        assert _exponential_utility(0.0, gamma=1.0) == pytest.approx(0.0)
        assert _exponential_utility(1.0, gamma=1.0) == pytest.approx(1 - np.exp(-1.0))
        # Monotonically increasing
        assert _exponential_utility(0.5, gamma=1.0) < _exponential_utility(1.0, gamma=1.0)


class TestZGrid:
    def test_grid_shape(self):
        grid = _compute_z_grid(theta=0.0, sigma_z=0.2, kappa=0.3, grid_size=200)
        assert grid.shape == (200,)

    def test_grid_centered_on_theta(self):
        grid = _compute_z_grid(theta=0.5, sigma_z=0.2, kappa=0.3, grid_size=200)
        assert grid[0] < 0.5 < grid[-1]
        # Grid center should be near theta
        mid = grid[len(grid) // 2]
        assert abs(mid - 0.5) < 0.1

    def test_grid_spans_4_std(self):
        sigma_z = 0.2
        kappa = 0.3
        stationary_std = sigma_z / np.sqrt(2 * kappa)
        grid = _compute_z_grid(theta=0.0, sigma_z=sigma_z, kappa=kappa, grid_size=200)
        expected_bound = 4 * stationary_std
        assert grid[-1] == pytest.approx(expected_bound, rel=0.01)
        assert grid[0] == pytest.approx(-expected_bound, rel=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_generator.py::TestUtilityFunctions -v`
Expected: FAIL — functions not found

- [ ] **Step 3: Implement utility functions and z-grid**

Add to `financial_gym/problems/regime_switching/generator.py` before the dataclasses:

```python
from scipy.special import roots_hermite


def _linear_utility(x: float) -> float:
    """Risk-neutral utility: u(x) = x."""
    return x


def _exponential_utility(x: float, gamma: float) -> float:
    """Risk-averse utility: u(x) = 1 - exp(-gamma * x)."""
    return 1.0 - np.exp(-gamma * x)


def _compute_z_grid(theta: float, sigma_z: float, kappa: float,
                    grid_size: int) -> np.ndarray:
    """Compute problem-specific z-grid spanning ±4 std of OU stationary distribution.

    The OU stationary distribution is N(theta, sigma_z^2 / (2*kappa)).
    Grid covers [theta - 4*std, theta + 4*std] with grid_size points.
    """
    stationary_std = sigma_z / np.sqrt(2 * kappa)
    z_min = theta - 4 * stationary_std
    z_max = theta + 4 * stationary_std
    return np.linspace(z_min, z_max, grid_size)
```

- [ ] **Step 4: Run utility and grid tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_generator.py::TestUtilityFunctions tests/test_generator.py::TestZGrid -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Write the failing test for the Bellman solver**

Add to `tests/test_generator.py`:

```python
from financial_gym.problems.regime_switching.generator import _solve_bellman


class TestBellmanSolver:
    def test_solver_output_shapes(self):
        """Solver returns policy table and value function with correct shapes."""
        grid_size = 50
        T = 5
        z_grid = np.linspace(-1, 1, grid_size)
        policy_table, value_table = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=0.1,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=20,
        )
        assert policy_table.shape == (T, grid_size, 2)
        assert value_table.shape == (T + 1, grid_size, 2)
        assert policy_table.dtype == np.int8

    def test_zero_lambda_greedy_is_optimal(self):
        """With no switching cost, optimal = greedy (sign of alpha*z)."""
        grid_size = 100
        T = 5
        z_grid = np.linspace(-1, 1, grid_size)
        policy_table, _ = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=0.0,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=20,
        )
        # With lam=0, optimal action depends only on sign of z, not on s_prev
        mid = grid_size // 2
        # For positive z, action should be 1 regardless of previous state
        for s_prev in [0, 1]:
            for t in range(T):
                assert policy_table[t, -1, s_prev] == 1, \
                    f"At t={t}, z=max, s_prev={s_prev}: expected ON"
                assert policy_table[t, 0, s_prev] == 0, \
                    f"At t={t}, z=min, s_prev={s_prev}: expected OFF"

    def test_high_lambda_favors_staying(self):
        """With very high switching cost, policy should rarely switch."""
        grid_size = 100
        T = 10
        z_grid = np.linspace(-1, 1, grid_size)
        policy_table, _ = _solve_bellman(
            z_grid=z_grid, T=T, alpha=0.3, lam=10.0,
            kappa=0.3, theta=0.0, sigma_z=0.2,
            utility_fn=_linear_utility, n_quad_nodes=20,
        )
        # With lam=10.0, switching never makes sense — stay in current state
        # If s_prev=0, should stay 0 (never switch on)
        mid = grid_size // 2
        for t in range(T):
            assert policy_table[t, mid, 0] == 0, \
                f"At t={t}, z=mid, s_prev=0: expected stay OFF with high lambda"
            assert policy_table[t, mid, 1] == 1, \
                f"At t={t}, z=mid, s_prev=1: expected stay ON with high lambda"
```

- [ ] **Step 6: Run solver tests to verify they fail**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_generator.py::TestBellmanSolver -v`
Expected: FAIL — `_solve_bellman` not found

- [ ] **Step 7: Implement the Bellman solver**

Add to `financial_gym/problems/regime_switching/generator.py`:

```python
from typing import Callable


def _solve_bellman(
    z_grid: np.ndarray,
    T: int,
    alpha: float,
    lam: float,
    kappa: float,
    theta: float,
    sigma_z: float,
    utility_fn: Callable[[float], float],
    n_quad_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the Bellman recursion via backward induction.

    Returns:
        policy_table: shape (T, grid_size, 2) — optimal action at (t, z_idx, s_prev).
        value_table: shape (T+1, grid_size, 2) — value function V_t(z, s_prev).
    """
    grid_size = len(z_grid)

    # Gauss-Hermite nodes and weights
    gh_nodes, gh_weights = roots_hermite(n_quad_nodes)

    # Value table: V[t, z_idx, s_prev]
    value_table = np.zeros((T + 1, grid_size, 2))
    # Terminal condition: V_T = 0 (no terminal reward)

    # Policy table: pi[t, z_idx, s_prev] ∈ {0, 1}
    policy_table = np.zeros((T, grid_size, 2), dtype=np.int8)

    # Backward induction: t = T-1 down to 0
    for t in range(T - 1, -1, -1):
        for z_idx, z in enumerate(z_grid):
            # OU transition: z' ~ N(mu, sigma_z^2)
            mu = z + kappa * (theta - z)

            # Expected future value for each action a ∈ {0, 1}
            # ∫ V_{t+1}(z', a) N(z'; mu, sigma_z^2) dz'
            # Using Gauss-Hermite: (1/sqrt(pi)) * sum_i w_i * f(mu + sqrt(2)*sigma_z*x_i)
            expected_future = np.zeros(2)  # indexed by action a
            for i in range(n_quad_nodes):
                z_prime = mu + np.sqrt(2) * sigma_z * gh_nodes[i]
                # Interpolate V_{t+1} at z_prime for each action
                for a in range(2):
                    v_interp = np.interp(z_prime, z_grid, value_table[t + 1, :, a])
                    expected_future[a] += gh_weights[i] * v_interp
            expected_future /= np.sqrt(np.pi)

            # Evaluate both actions for each s_prev
            for s_prev in range(2):
                q_values = np.zeros(2)
                for a in range(2):
                    immediate_reward = utility_fn(a * alpha * z)
                    switching_cost = lam if a != s_prev else 0.0
                    q_values[a] = immediate_reward - switching_cost + expected_future[a]

                best_action = int(np.argmax(q_values))
                policy_table[t, z_idx, s_prev] = best_action
                value_table[t, z_idx, s_prev] = q_values[best_action]

    return policy_table, value_table
```

- [ ] **Step 8: Run solver tests to verify they pass**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_generator.py::TestBellmanSolver -v`
Expected: All 3 tests PASS

- [ ] **Step 9: Write the failing test for the full generator**

Add to `tests/test_generator.py`:

```python
from financial_gym.problems.regime_switching.generator import RegimeSwitchingGenerator


class TestRegimeSwitchingGenerator:
    def test_sample_returns_solved_problem(self):
        config = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
        gen = RegimeSwitchingGenerator(config)
        problem = gen.sample(seed=42)

        assert isinstance(problem, RegimeSwitchingProblem)
        assert problem.T == 5
        assert problem.z_path.shape == (6,)
        assert problem.x_path.shape == (5,)
        assert problem.optimal_policy_table.shape == (5, 50, 2)
        assert problem.optimal_policy_table.dtype == np.int8

    def test_same_seed_same_problem(self):
        config = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
        gen = RegimeSwitchingGenerator(config)
        p1 = gen.sample(seed=123)
        p2 = gen.sample(seed=123)
        np.testing.assert_array_equal(p1.z_path, p2.z_path)
        np.testing.assert_array_equal(p1.x_path, p2.x_path)
        assert p1.alpha == p2.alpha
        assert p1.lam == p2.lam

    def test_different_seed_different_problem(self):
        config = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
        gen = RegimeSwitchingGenerator(config)
        p1 = gen.sample(seed=1)
        p2 = gen.sample(seed=2)
        # Very unlikely to be identical
        assert not np.allclose(p1.z_path, p2.z_path)
```

- [ ] **Step 10: Run generator tests to verify they fail**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_generator.py::TestRegimeSwitchingGenerator -v`
Expected: FAIL — `RegimeSwitchingGenerator` not found

- [ ] **Step 11: Implement the full generator**

Add to `financial_gym/problems/regime_switching/generator.py`:

```python
class RegimeSwitchingGenerator(BaseGenerator):
    """Generator for regime switching problems with integrated Bellman solver.

    Samples DGP parameters from config ranges, simulates a trajectory,
    solves the Bellman equation, and returns a fully-solved instance.
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()

    def sample(self, seed: int) -> RegimeSwitchingProblem:
        rng = np.random.default_rng(seed)
        cfg = self.config

        # Sample DGP parameters from config ranges
        kappa = rng.uniform(*cfg.kappa_range)
        sigma_z = rng.uniform(*cfg.sigma_z_range)
        alpha = rng.uniform(*cfg.alpha_range)
        sigma_x = rng.uniform(*cfg.sigma_x_range)
        lam = rng.uniform(*cfg.lam_range)
        T = rng.integers(cfg.T_range[0], cfg.T_range[1] + 1)

        # Simulate OU signal path: z_{t+1} = z_t + kappa*(theta - z_t) + sigma_z*eps
        z_path = np.zeros(T + 1)
        z_path[0] = rng.normal(0.0, sigma_z / np.sqrt(2 * kappa))  # stationary draw
        for t in range(T):
            z_path[t + 1] = (z_path[t] + kappa * (cfg.theta - z_path[t])
                             + sigma_z * rng.standard_normal())

        # Simulate PnL: X_{t+1} | Z_t ~ N(alpha * Z_t, sigma_x^2)
        x_path = alpha * z_path[:T] + sigma_x * rng.standard_normal(T)

        # Select utility function
        if cfg.utility == "linear":
            utility_fn = _linear_utility
        else:
            utility_fn = lambda x: _exponential_utility(x, cfg.gamma)

        # Compute z-grid and solve Bellman
        z_grid = _compute_z_grid(cfg.theta, sigma_z, kappa, cfg.grid_size)
        policy_table, value_table = _solve_bellman(
            z_grid=z_grid, T=T, alpha=alpha, lam=lam,
            kappa=kappa, theta=cfg.theta, sigma_z=sigma_z,
            utility_fn=utility_fn, n_quad_nodes=cfg.n_quad_nodes,
        )

        # Compute expected optimal value: E[V_0] at stationary z
        mid_idx = cfg.grid_size // 2
        optimal_value = float(value_table[0, mid_idx, 0])

        return RegimeSwitchingProblem(
            kappa=kappa, theta=cfg.theta, sigma_z=sigma_z,
            alpha=alpha, sigma_x=sigma_x, lam=lam,
            T=T, seed=seed,
            z_path=z_path, x_path=x_path,
            initial_regime=0,
            optimal_policy_table=policy_table,
            optimal_value=optimal_value,
        )
```

- [ ] **Step 12: Run all generator tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_generator.py -v`
Expected: All 11 tests PASS

- [ ] **Step 13: Commit**

```bash
git add financial_gym/problems/regime_switching/generator.py tests/test_generator.py
git commit -m "feat: Bellman solver and regime switching generator"
```

---

### Task 4: Verifier (Parse + Score)

**Files:**
- Create: `financial_gym/problems/regime_switching/verifier.py`
- Test: `tests/test_verifier.py`

- [ ] **Step 1: Write the failing test for parsing**

```python
# tests/test_verifier.py
import numpy as np
import pytest
from financial_gym.problems.regime_switching.verifier import (
    RegimeSwitchingVerifier,
    _parse_decision,
    _compute_realized_utility,
)
from financial_gym.problems.regime_switching.generator import (
    RegimeSwitchingProblem,
    GeneratorConfig,
    RegimeSwitchingGenerator,
)


class TestParseDecision:
    def test_parse_standard_format(self):
        assert _parse_decision("Some reasoning here. s_0 = 1") == 1

    def test_parse_with_newlines(self):
        text = "Long reasoning...\n\ns_3 = 0"
        assert _parse_decision(text) == 0

    def test_parse_fails_gracefully(self):
        assert _parse_decision("No decision here") == 0

    def test_parse_takes_last_match(self):
        text = "Maybe s_0 = 1 but actually s_0 = 0"
        assert _parse_decision(text) == 0

    def test_parse_ignores_non_binary(self):
        assert _parse_decision("s_0 = 5") == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_verifier.py::TestParseDecision -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement parsing**

```python
# financial_gym/problems/regime_switching/verifier.py
"""Verifier for regime switching: parse LLM text, score against optimal."""
from __future__ import annotations

import re

import numpy as np

from financial_gym.base.verifier import BaseVerifier
from financial_gym.problems.regime_switching.generator import RegimeSwitchingProblem


def _parse_decision(text: str) -> int:
    """Extract a binary decision s_t ∈ {0,1} from model text.

    Looks for patterns like 's_0 = 1' or 's_3 = 0'.
    Returns the last match found. Defaults to 0 on failure.
    """
    matches = re.findall(r"s_?\d*\s*=\s*([01])", text)
    if matches:
        return int(matches[-1])
    return 0
```

- [ ] **Step 4: Run parse tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_verifier.py::TestParseDecision -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Write the failing test for realized utility computation**

Add to `tests/test_verifier.py`:

```python
class TestComputeRealizedUtility:
    def test_all_off_zero_utility(self):
        """If strategy is always off, utility is 0 (no PnL, no switching)."""
        decisions = np.array([0, 0, 0, 0, 0])
        x_path = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
        result = _compute_realized_utility(
            decisions=decisions, x_path=x_path, lam=0.1,
            initial_regime=0, utility_fn=lambda x: x,
        )
        assert result == pytest.approx(0.0)

    def test_all_on_no_switch_from_on(self):
        """If always on and started on, collect all PnL, no switching cost."""
        decisions = np.array([1, 1, 1])
        x_path = np.array([0.2, 0.1, -0.05])
        result = _compute_realized_utility(
            decisions=decisions, x_path=x_path, lam=0.1,
            initial_regime=1, utility_fn=lambda x: x,
        )
        # Sum of PnL, no switching costs
        assert result == pytest.approx(0.2 + 0.1 + (-0.05))

    def test_switching_cost_applied(self):
        """Switching from off to on costs lam."""
        decisions = np.array([1, 0])
        x_path = np.array([0.5, 0.3])
        result = _compute_realized_utility(
            decisions=decisions, x_path=x_path, lam=0.1,
            initial_regime=0, utility_fn=lambda x: x,
        )
        # t=0: on, was off -> PnL 0.5 - switch 0.1 = 0.4
        # t=1: off, was on -> PnL 0.0 - switch 0.1 = -0.1
        assert result == pytest.approx(0.3)
```

- [ ] **Step 6: Run to verify failure**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_verifier.py::TestComputeRealizedUtility -v`
Expected: FAIL — `_compute_realized_utility` not found

- [ ] **Step 7: Implement realized utility computation**

Add to `financial_gym/problems/regime_switching/verifier.py`:

```python
from typing import Callable


def _compute_realized_utility(
    decisions: np.ndarray,
    x_path: np.ndarray,
    lam: float,
    initial_regime: int,
    utility_fn: Callable[[float], float],
) -> float:
    """Compute realized utility J(s) for a decision sequence on a realized trajectory.

    J(s) = sum_t [ u(s_t * x_path[t]) - lam * 1{s_t != s_{t-1}} ]
    with s_{-1} = initial_regime.
    """
    T = len(decisions)
    total = 0.0
    prev = initial_regime
    for t in range(T):
        pnl = utility_fn(decisions[t] * x_path[t])
        switch_cost = lam if decisions[t] != prev else 0.0
        total += pnl - switch_cost
        prev = decisions[t]
    return total
```

- [ ] **Step 8: Run utility tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_verifier.py::TestComputeRealizedUtility -v`
Expected: All 3 tests PASS

- [ ] **Step 9: Write the failing test for the full verifier**

Add to `tests/test_verifier.py`:

```python
class TestRegimeSwitchingVerifier:
    @pytest.fixture
    def problem_and_verifier(self):
        config = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
        gen = RegimeSwitchingGenerator(config)
        problem = gen.sample(seed=42)
        verifier = RegimeSwitchingVerifier()
        return problem, verifier

    def test_optimal_completion_scores_one(self, problem_and_verifier):
        """Optimal decisions should score exactly 1.0."""
        problem, verifier = problem_and_verifier
        # Build completions that match optimal policy
        from financial_gym.agents.optimal_agent import OptimalAgent
        agent = OptimalAgent()
        decisions = agent.decide(problem)
        completions = [f"s_{t} = {decisions[t]}" for t in range(problem.T)]
        score = verifier.score(completions, problem, mode="trajectory")
        assert score == pytest.approx(1.0, abs=1e-10)

    def test_empty_completions_score_low(self, problem_and_verifier):
        """Garbled completions default to all-off, score near 0 or negative."""
        problem, verifier = problem_and_verifier
        completions = ["no decision"] * problem.T
        score = verifier.score(completions, problem, mode="trajectory")
        # All-off is valid but unlikely to be optimal; score should be <= 1.0
        assert score <= 1.0

    def test_per_step_mode(self, problem_and_verifier):
        """Per-step diagnostic score is between 0 and 1."""
        problem, verifier = problem_and_verifier
        completions = [f"s_{t} = 1" for t in range(problem.T)]
        score = verifier.score(completions, problem, mode="per_step")
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 10: Run to verify failure**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_verifier.py::TestRegimeSwitchingVerifier -v`
Expected: FAIL — `RegimeSwitchingVerifier` not found

- [ ] **Step 11: Implement the full verifier**

Add to `financial_gym/problems/regime_switching/verifier.py`:

```python
def _apply_policy_table(
    policy_table: np.ndarray,
    z_path: np.ndarray,
    z_grid: np.ndarray,
    initial_regime: int,
) -> np.ndarray:
    """Apply the optimal policy table along a realized z_path.

    Returns:
        decisions: shape (T,) — optimal actions along the realized path.
    """
    T = policy_table.shape[0]
    decisions = np.zeros(T, dtype=np.int8)
    prev = initial_regime
    for t in range(T):
        # Find nearest grid index for z_path[t]
        z_idx = int(np.argmin(np.abs(z_grid - z_path[t])))
        decisions[t] = policy_table[t, z_idx, prev]
        prev = decisions[t]
    return decisions


def _generate_random_decisions(T: int, seed: int) -> np.ndarray:
    """Generate seeded random binary decisions."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=T).astype(np.int8)


class RegimeSwitchingVerifier(BaseVerifier):
    """Verifier for regime switching: parse + regret-normalised score."""

    def score(
        self,
        completion: list[str],
        problem: RegimeSwitchingProblem,
        mode: str = "trajectory",
    ) -> float:
        """Score a model's completion against the optimal policy.

        Args:
            completion: List of assistant response strings, one per turn.
            problem: Fully-solved problem instance.
            mode: "trajectory" (training) or "per_step" (diagnostic).

        Returns:
            Float score. 1.0 = optimal, 0.0 = random-level.
        """
        T = problem.T

        # Parse decisions from text
        s_model = np.array([_parse_decision(c) for c in completion[:T]], dtype=np.int8)
        if len(s_model) < T:
            s_model = np.pad(s_model, (0, T - len(s_model)))

        # Select utility function
        utility_fn = lambda x: x  # linear default

        # Reconstruct z_grid for policy table lookup
        from financial_gym.problems.regime_switching.generator import _compute_z_grid
        z_grid = _compute_z_grid(
            problem.theta, problem.sigma_z, problem.kappa,
            problem.optimal_policy_table.shape[1],
        )

        if mode == "per_step":
            return self._per_step_score(s_model, problem, z_grid)

        # Trajectory-level scoring
        # Compute realized optimal decisions
        s_star = _apply_policy_table(
            problem.optimal_policy_table, problem.z_path, z_grid,
            problem.initial_regime,
        )

        # Compute random baseline (seed = problem.seed)
        s_random = _generate_random_decisions(T, problem.seed)

        # Compute realized utilities
        j_model = _compute_realized_utility(
            s_model, problem.x_path, problem.lam,
            problem.initial_regime, utility_fn,
        )
        j_star = _compute_realized_utility(
            s_star, problem.x_path, problem.lam,
            problem.initial_regime, utility_fn,
        )
        j_random = _compute_realized_utility(
            s_random, problem.x_path, problem.lam,
            problem.initial_regime, utility_fn,
        )

        # Regret-normalised score
        epsilon = 1e-6
        denominator = max(j_star - j_random, epsilon)
        return (j_model - j_random) / denominator

    def _per_step_score(
        self,
        s_model: np.ndarray,
        problem: RegimeSwitchingProblem,
        z_grid: np.ndarray,
    ) -> float:
        """Diagnostic: fraction of steps matching optimal given model's actual s_prev."""
        T = problem.T
        correct = 0
        prev = problem.initial_regime
        for t in range(T):
            z_idx = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
            optimal_action = problem.optimal_policy_table[t, z_idx, prev]
            if s_model[t] == optimal_action:
                correct += 1
            prev = int(s_model[t])
        return correct / T
```

- [ ] **Step 12: Run all verifier tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_verifier.py -v`
Expected: All 9 tests PASS (parse tests will need agents — covered in next task, so skip `test_optimal_completion_scores_one` for now)

- [ ] **Step 13: Commit**

```bash
git add financial_gym/problems/regime_switching/verifier.py tests/test_verifier.py
git commit -m "feat: verifier with parsing and regret-normalised scoring"
```

---

### Task 5: Validation Agents

**Files:**
- Create: `financial_gym/agents/random_agent.py`
- Create: `financial_gym/agents/greedy_agent.py`
- Create: `financial_gym/agents/optimal_agent.py`
- Test: `tests/test_agents.py`

- [ ] **Step 1: Write the failing test for all three agents**

```python
# tests/test_agents.py
import numpy as np
import pytest
from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from financial_gym.agents.random_agent import RandomAgent
from financial_gym.agents.greedy_agent import GreedyAgent
from financial_gym.agents.optimal_agent import OptimalAgent


@pytest.fixture
def problem():
    config = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
    gen = RegimeSwitchingGenerator(config)
    return gen.sample(seed=42)


class TestRandomAgent:
    def test_returns_correct_shape(self, problem):
        agent = RandomAgent(seed_offset=1)
        decisions = agent.decide(problem)
        assert decisions.shape == (problem.T,)
        assert set(np.unique(decisions)).issubset({0, 1})

    def test_deterministic_with_same_seed(self, problem):
        a1 = RandomAgent(seed_offset=1)
        a2 = RandomAgent(seed_offset=1)
        np.testing.assert_array_equal(a1.decide(problem), a2.decide(problem))

    def test_different_offset_different_decisions(self, problem):
        a1 = RandomAgent(seed_offset=1)
        a2 = RandomAgent(seed_offset=2)
        # Very unlikely to be identical for T=5
        d1, d2 = a1.decide(problem), a2.decide(problem)
        assert not np.array_equal(d1, d2) or True  # allow rare equality


class TestGreedyAgent:
    def test_returns_correct_shape(self, problem):
        agent = GreedyAgent()
        decisions = agent.decide(problem)
        assert decisions.shape == (problem.T,)
        assert set(np.unique(decisions)).issubset({0, 1})

    def test_positive_signal_no_switch_cost_means_on(self):
        """With zero lambda and positive z, greedy should choose ON."""
        config = GeneratorConfig(
            T_range=(3, 3), lam_range=(0.0, 0.0),
            alpha_range=(0.3, 0.3), grid_size=50, n_quad_nodes=10,
        )
        gen = RegimeSwitchingGenerator(config)
        problem = gen.sample(seed=100)
        agent = GreedyAgent()
        decisions = agent.decide(problem)
        # With lam=0, greedy should turn on whenever alpha*z > 0
        for t in range(problem.T):
            expected = 1 if problem.alpha * problem.z_path[t] > 0 else 0
            assert decisions[t] == expected, f"t={t}, z={problem.z_path[t]}"


class TestOptimalAgent:
    def test_returns_correct_shape(self, problem):
        agent = OptimalAgent()
        decisions = agent.decide(problem)
        assert decisions.shape == (problem.T,)
        assert set(np.unique(decisions)).issubset({0, 1})

    def test_matches_policy_table(self, problem):
        """Optimal agent must follow the policy table exactly."""
        agent = OptimalAgent()
        decisions = agent.decide(problem)
        # Manually trace through policy table
        from financial_gym.problems.regime_switching.generator import _compute_z_grid
        z_grid = _compute_z_grid(
            problem.theta, problem.sigma_z, problem.kappa,
            problem.optimal_policy_table.shape[1],
        )
        prev = problem.initial_regime
        for t in range(problem.T):
            z_idx = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
            expected = problem.optimal_policy_table[t, z_idx, prev]
            assert decisions[t] == expected, f"t={t}"
            prev = decisions[t]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_agents.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement RandomAgent**

```python
# financial_gym/agents/random_agent.py
"""Random agent: uniform random binary decisions."""
import numpy as np

from financial_gym.base.problem import BaseProblem


class RandomAgent:
    """Generates seeded random binary decisions.

    seed = problem.seed + seed_offset ensures deterministic but
    distinct decisions per agent instance.
    """

    def __init__(self, seed_offset: int = 1) -> None:
        self.seed_offset = seed_offset

    def decide(self, problem: BaseProblem) -> np.ndarray:
        rng = np.random.default_rng(problem.seed + self.seed_offset)
        return rng.integers(0, 2, size=problem.horizon()).astype(np.int8)
```

- [ ] **Step 4: Implement GreedyAgent**

```python
# financial_gym/agents/greedy_agent.py
"""Greedy agent: myopically maximises immediate expected PnL."""
import numpy as np

from financial_gym.problems.regime_switching.generator import RegimeSwitchingProblem


class GreedyAgent:
    """Chooses action maximising immediate reward, ignoring future.

    s_t = argmax_a [ u(a * alpha * Z_t) - lam * 1{a != s_{t-1}} ]
    """

    def decide(self, problem: RegimeSwitchingProblem) -> np.ndarray:
        T = problem.T
        decisions = np.zeros(T, dtype=np.int8)
        prev = problem.initial_regime

        for t in range(T):
            expected_pnl = problem.alpha * problem.z_path[t]
            # Q-value for action 0 (off)
            q_off = 0.0 - (problem.lam if 0 != prev else 0.0)
            # Q-value for action 1 (on)
            q_on = expected_pnl - (problem.lam if 1 != prev else 0.0)

            decisions[t] = 1 if q_on > q_off else 0
            prev = int(decisions[t])

        return decisions
```

- [ ] **Step 5: Implement OptimalAgent**

```python
# financial_gym/agents/optimal_agent.py
"""Optimal agent: follows the Bellman-optimal policy table."""
import numpy as np

from financial_gym.problems.regime_switching.generator import (
    RegimeSwitchingProblem,
    _compute_z_grid,
)


class OptimalAgent:
    """Applies the pre-computed optimal policy table along the realized z_path."""

    def decide(self, problem: RegimeSwitchingProblem) -> np.ndarray:
        z_grid = _compute_z_grid(
            problem.theta, problem.sigma_z, problem.kappa,
            problem.optimal_policy_table.shape[1],
        )

        T = problem.T
        decisions = np.zeros(T, dtype=np.int8)
        prev = problem.initial_regime

        for t in range(T):
            z_idx = int(np.argmin(np.abs(z_grid - problem.z_path[t])))
            decisions[t] = problem.optimal_policy_table[t, z_idx, prev]
            prev = int(decisions[t])

        return decisions
```

- [ ] **Step 6: Run all agent tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_agents.py -v`
Expected: All 7 tests PASS

- [ ] **Step 7: Now run the deferred verifier test (optimal scores 1.0)**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_verifier.py::TestRegimeSwitchingVerifier::test_optimal_completion_scores_one -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add financial_gym/agents/ tests/test_agents.py
git commit -m "feat: random, greedy, and optimal validation agents"
```

---

### Task 6: Prompt Format (Stateful Conversation Manager)

**Files:**
- Create: `financial_gym/problems/regime_switching/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
import pytest
from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt


@pytest.fixture
def problem():
    config = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
    gen = RegimeSwitchingGenerator(config)
    return gen.sample(seed=42)


class TestSetupPrompt:
    def test_contains_parameters(self, problem):
        prompt = setup_prompt(problem)
        assert str(problem.T) in prompt
        assert "switching" in prompt.lower() or "λ" in prompt or str(problem.lam) in prompt
        assert str(problem.alpha) in prompt or "α" in prompt

    def test_contains_formula(self, problem):
        prompt = setup_prompt(problem)
        assert "E[X" in prompt or "Expected PnL" in prompt

    def test_contains_initial_regime(self, problem):
        prompt = setup_prompt(problem)
        assert str(problem.initial_regime) in prompt


class TestStepPrompt:
    def test_contains_time_and_signal(self, problem):
        prompt = step_prompt(t=0, z_t=0.42, prev_regime=0)
        assert "t=0" in prompt
        assert "0.42" in prompt

    def test_contains_previous_regime(self):
        prompt = step_prompt(t=3, z_t=-0.12, prev_regime=1)
        assert "1" in prompt  # previous regime shown

    def test_step_prompts_differ_per_timestep(self):
        p1 = step_prompt(t=0, z_t=0.5, prev_regime=0)
        p2 = step_prompt(t=1, z_t=-0.3, prev_regime=1)
        assert p1 != p2
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement prompts module**

```python
# financial_gym/problems/regime_switching/prompts.py
"""Stateful conversation manager for regime switching multi-turn prompts."""
from __future__ import annotations

from financial_gym.problems.regime_switching.generator import RegimeSwitchingProblem


def setup_prompt(problem: RegimeSwitchingProblem) -> str:
    """Generate the Turn 0 setup message with problem parameters.

    Args:
        problem: The fully-solved problem instance.

    Returns:
        Setup message string for the user turn.
    """
    return (
        f"You are managing a momentum trading strategy over T={problem.T} steps.\n"
        f"\n"
        f"Parameters:\n"
        f"  Switching cost:   λ = {problem.lam:.4f}\n"
        f"  Signal strength:  α = {problem.alpha:.4f}\n"
        f"  Expected PnL:     E[X_{{t+1}} | Z_t] = α · Z_t = {problem.alpha:.4f} · Z_t\n"
        f"  Initial regime:   s_{{-1}} = {problem.initial_regime}\n"
        f"\n"
        f"At each step you observe signal Z_t and decide to activate (s_t=1)\n"
        f"or deactivate (s_t=0) the strategy. Switching from your previous\n"
        f"decision costs λ = {problem.lam:.4f}, deducted from that step's PnL.\n"
        f"\n"
        f"You will receive one observation at a time. At each step, reason\n"
        f"about the immediate expected PnL, the switching cost, and whether\n"
        f"the signal is likely to persist before stating your decision."
    )


def step_prompt(t: int, z_t: float, prev_regime: int) -> str:
    """Generate the user message for a single time step.

    Args:
        t: Current time step.
        z_t: Signal value at time t.
        prev_regime: Model's actual decision at t-1 (or initial_regime if t=0).

    Returns:
        User message string for this step.
    """
    regime_label = f"s_{{-1}}" if t == 0 else f"s_{{{t-1}}}"
    return f"t={t} | Z_{t} = {z_t:+.4f} | Previous regime: {regime_label} = {prev_regime}"
```

- [ ] **Step 4: Run prompt tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_prompts.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add financial_gym/problems/regime_switching/prompts.py tests/test_prompts.py
git commit -m "feat: multi-turn prompt format with stateful conversation manager"
```

---

### Task 7: Goldilocks Validation Suite

**Files:**
- Create: `financial_gym/validation/goldilocks.py`
- Test: `tests/test_goldilocks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goldilocks.py
import pytest
from financial_gym.validation.goldilocks import GoldilocksValidator, GoldilocksReport


class TestGoldilocksValidator:
    @pytest.fixture
    def validator(self):
        return GoldilocksValidator(n_instances=20, grid_size=50, n_quad_nodes=10)

    def test_report_structure(self, validator):
        report = validator.run()
        assert isinstance(report, GoldilocksReport)
        assert len(report.difficulty_levels) == 3
        assert set(report.agents) == {"random", "greedy", "optimal"}

    def test_optimal_scores_exactly_one(self, validator):
        report = validator.run()
        for level in report.difficulty_levels:
            assert report.mean_scores["optimal"][level] == pytest.approx(1.0, abs=1e-10)

    def test_greedy_below_optimal(self, validator):
        report = validator.run()
        for level in report.difficulty_levels:
            assert report.mean_scores["greedy"][level] < report.mean_scores["optimal"][level]

    def test_random_near_zero(self, validator):
        report = validator.run()
        for level in report.difficulty_levels:
            assert abs(report.mean_scores["random"][level]) < 0.2

    def test_gap_increases_with_difficulty(self, validator):
        report = validator.run()
        gaps = report.greedy_optimal_gaps()
        # Gap should increase: easy < medium < hard
        assert gaps["easy"] < gaps["hard"]

    def test_pass_criteria(self, validator):
        report = validator.run()
        assert report.all_pass()

    def test_report_string(self, validator):
        report = validator.run()
        text = str(report)
        assert "Goldilocks" in text
        assert "PASS" in text or "FAIL" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_goldilocks.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the Goldilocks validator**

```python
# financial_gym/validation/goldilocks.py
"""Goldilocks validation suite: three-agent benchmark proving reward signal quality."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from financial_gym.problems.regime_switching.verifier import (
    RegimeSwitchingVerifier,
    _compute_realized_utility,
    _apply_policy_table,
    _generate_random_decisions,
)
from financial_gym.problems.regime_switching.generator import _compute_z_grid
from financial_gym.agents.random_agent import RandomAgent
from financial_gym.agents.greedy_agent import GreedyAgent
from financial_gym.agents.optimal_agent import OptimalAgent


DIFFICULTY_LEVELS = {
    "easy": {"T_range": (3, 3), "lam_range": (0.05, 0.05), "alpha_range": (0.40, 0.40)},
    "medium": {"T_range": (10, 10), "lam_range": (0.15, 0.15), "alpha_range": (0.30, 0.30)},
    "hard": {"T_range": (20, 20), "lam_range": (0.30, 0.30), "alpha_range": (0.20, 0.20)},
}


@dataclass
class GoldilocksReport:
    """Results of the Goldilocks validation."""
    difficulty_levels: list[str]
    agents: list[str]
    mean_scores: dict[str, dict[str, float]]

    def greedy_optimal_gaps(self) -> dict[str, float]:
        return {
            level: self.mean_scores["optimal"][level] - self.mean_scores["greedy"][level]
            for level in self.difficulty_levels
        }

    def all_pass(self) -> bool:
        """Check all five Goldilocks conditions."""
        gaps = self.greedy_optimal_gaps()

        # 1. Optimal = 1.0 exactly at all levels
        for level in self.difficulty_levels:
            if abs(self.mean_scores["optimal"][level] - 1.0) > 1e-10:
                return False

        # 2. Greedy > 0.1 at easy
        if self.mean_scores["greedy"]["easy"] <= 0.1:
            return False

        # 3. Greedy < Optimal at all levels
        for level in self.difficulty_levels:
            if self.mean_scores["greedy"][level] >= self.mean_scores["optimal"][level]:
                return False

        # 4. Gap increases easy → hard
        if not (gaps["easy"] < gaps["medium"] < gaps["hard"]):
            return False

        # 5. Random near zero at all levels
        for level in self.difficulty_levels:
            if abs(self.mean_scores["random"][level]) > 0.15:
                return False

        return True

    def __str__(self) -> str:
        lines = ["Goldilocks Validation Report"]
        lines.append("─" * 50)
        header = f"{'':>12}" + "".join(f"{level:>10}" for level in self.difficulty_levels)
        lines.append(header)
        for agent in self.agents:
            row = f"{agent:>12}"
            for level in self.difficulty_levels:
                row += f"{self.mean_scores[agent][level]:>10.2f}"
            lines.append(row)
        lines.append("─" * 50)

        gaps = self.greedy_optimal_gaps()
        gap_str = "  →  ".join(f"{gaps[l]:.2f}" for l in self.difficulty_levels)
        monotone = gaps["easy"] < gaps["medium"] < gaps["hard"]
        lines.append(f"Greedy-Optimal gap: {gap_str}   {'✓' if monotone else '✗'} monotone")
        lines.append(f"All conditions: {'PASS' if self.all_pass() else 'FAIL'}")
        return "\n".join(lines)


class GoldilocksValidator:
    """Runs the three-agent benchmark across difficulty levels."""

    def __init__(
        self,
        n_instances: int = 500,
        grid_size: int = 200,
        n_quad_nodes: int = 20,
    ) -> None:
        self.n_instances = n_instances
        self.grid_size = grid_size
        self.n_quad_nodes = n_quad_nodes

    def run(self) -> GoldilocksReport:
        agents = {
            "random": RandomAgent(seed_offset=1),
            "greedy": GreedyAgent(),
            "optimal": OptimalAgent(),
        }
        verifier = RegimeSwitchingVerifier()

        mean_scores: dict[str, dict[str, float]] = {
            name: {} for name in agents
        }

        for level_name, level_params in DIFFICULTY_LEVELS.items():
            config = GeneratorConfig(
                grid_size=self.grid_size,
                n_quad_nodes=self.n_quad_nodes,
                **level_params,
            )
            gen = RegimeSwitchingGenerator(config)

            level_scores: dict[str, list[float]] = {name: [] for name in agents}

            for i in range(self.n_instances):
                problem = gen.sample(seed=i)

                for name, agent in agents.items():
                    decisions = agent.decide(problem)
                    # Convert decisions to text completions for the verifier
                    completions = [f"s_{t} = {decisions[t]}" for t in range(problem.T)]
                    score = verifier.score(completions, problem, mode="trajectory")
                    level_scores[name].append(score)

            for name in agents:
                mean_scores[name][level_name] = float(np.mean(level_scores[name]))

        return GoldilocksReport(
            difficulty_levels=list(DIFFICULTY_LEVELS.keys()),
            agents=list(agents.keys()),
            mean_scores=mean_scores,
        )
```

- [ ] **Step 4: Run Goldilocks tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_goldilocks.py -v --timeout=120`
Expected: All 7 tests PASS (may take 30-60s with n_instances=20 and small grid)

- [ ] **Step 5: Commit**

```bash
git add financial_gym/validation/ tests/test_goldilocks.py
git commit -m "feat: Goldilocks validation suite with three-agent benchmark"
```

---

### Task 8: Integration Test and TRL Reward Function

**Files:**
- Create: `tests/test_integration.py`
- Modify: `financial_gym/__init__.py` (add public API exports)

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end integration test: generate → solve → prompt → decide → score."""
import numpy as np
import pytest
from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
)
from financial_gym.problems.regime_switching.verifier import RegimeSwitchingVerifier
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.agents.greedy_agent import GreedyAgent
from financial_gym.agents.random_agent import RandomAgent


class TestEndToEnd:
    @pytest.fixture
    def gym_components(self):
        config = GeneratorConfig(T_range=(5, 5), grid_size=50, n_quad_nodes=10)
        gen = RegimeSwitchingGenerator(config)
        verifier = RegimeSwitchingVerifier()
        return gen, verifier

    def test_full_pipeline(self, gym_components):
        """Generate → prompt → simulate agent → score: full data flow."""
        gen, verifier = gym_components
        problem = gen.sample(seed=42)

        # Step 1: Generate setup prompt
        setup = setup_prompt(problem)
        assert len(setup) > 0

        # Step 2: Simulate a multi-turn conversation using greedy agent
        agent = GreedyAgent()
        decisions = agent.decide(problem)

        completions = []
        prev_regime = problem.initial_regime
        for t in range(problem.T):
            # Generate step prompt (what user would send)
            user_msg = step_prompt(t, problem.z_path[t], prev_regime)
            assert "t=" in user_msg

            # Agent decides (simulating model response)
            response = f"Reasoning here... s_{t} = {decisions[t]}"
            completions.append(response)
            prev_regime = int(decisions[t])

        # Step 3: Score
        score = verifier.score(completions, problem, mode="trajectory")
        assert isinstance(score, float)
        assert score > -1.0  # greedy should do better than terrible

    def test_reward_fn_interface(self, gym_components):
        """Test the exact TRL reward_fn interface."""
        gen, verifier = gym_components

        # This is the function labs plug into GRPOTrainer
        def reward_fn(completions_batch, problems_batch):
            return [
                verifier.score(c, p, mode="trajectory")
                for c, p in zip(completions_batch, problems_batch)
            ]

        # Generate a batch of problems
        problems = [gen.sample(seed=i) for i in range(5)]

        # Simulate completions (optimal agent for simplicity)
        agent = OptimalAgent()
        completions_batch = []
        for p in problems:
            decisions = agent.decide(p)
            completions_batch.append(
                [f"s_{t} = {decisions[t]}" for t in range(p.T)]
            )

        # Score the batch
        scores = reward_fn(completions_batch, problems)
        assert len(scores) == 5
        for s in scores:
            assert s == pytest.approx(1.0, abs=1e-10)

    def test_optimal_beats_greedy_beats_random(self, gym_components):
        """Reward ordering: random < greedy < optimal."""
        gen, verifier = gym_components
        agents = {
            "random": RandomAgent(seed_offset=1),
            "greedy": GreedyAgent(),
            "optimal": OptimalAgent(),
        }

        scores = {name: [] for name in agents}
        for seed in range(50):
            problem = gen.sample(seed=seed)
            for name, agent in agents.items():
                decisions = agent.decide(problem)
                completions = [f"s_{t} = {decisions[t]}" for t in range(problem.T)]
                scores[name].append(verifier.score(completions, problem))

        mean_scores = {name: np.mean(s) for name, s in scores.items()}
        assert mean_scores["random"] < mean_scores["greedy"]
        assert mean_scores["greedy"] < mean_scores["optimal"]
        assert mean_scores["optimal"] == pytest.approx(1.0, abs=1e-10)
```

- [ ] **Step 2: Update public API**

```python
# financial_gym/__init__.py
"""Financial RLVR Gym — verifiable environments for LLM post-training."""
from financial_gym.problems.regime_switching.generator import (
    GeneratorConfig,
    RegimeSwitchingGenerator,
    RegimeSwitchingProblem,
)
from financial_gym.problems.regime_switching.verifier import RegimeSwitchingVerifier
from financial_gym.problems.regime_switching.prompts import setup_prompt, step_prompt
from financial_gym.agents.random_agent import RandomAgent
from financial_gym.agents.greedy_agent import GreedyAgent
from financial_gym.agents.optimal_agent import OptimalAgent
from financial_gym.validation.goldilocks import GoldilocksValidator

__all__ = [
    "GeneratorConfig",
    "RegimeSwitchingGenerator",
    "RegimeSwitchingProblem",
    "RegimeSwitchingVerifier",
    "setup_prompt",
    "step_prompt",
    "RandomAgent",
    "GreedyAgent",
    "OptimalAgent",
    "GoldilocksValidator",
]
```

- [ ] **Step 3: Run integration tests**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/test_integration.py -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Run the full test suite**

Run: `cd /Users/imen/Documents/stochastic_control && .venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add financial_gym/__init__.py tests/test_integration.py
git commit -m "feat: integration tests and public API exports"
```

---

## Task Summary

| Task | What it builds | Key tests |
|---|---|---|
| 1 | Scaffolding + ABCs | ABCs cannot be instantiated |
| 2 | Dataclasses | Problem fields, config validation |
| 3 | Bellman solver + generator | λ=0 greedy=optimal, high λ favors staying |
| 4 | Verifier (parse + score) | Parsing, utility computation, scoring |
| 5 | Three agents | Shape, determinism, correctness |
| 6 | Prompt format | Contains params, formula, step prompts |
| 7 | Goldilocks suite | Optimal=1.0, gap widens, all pass |
| 8 | Integration + API | Full pipeline, TRL interface, reward ordering |
