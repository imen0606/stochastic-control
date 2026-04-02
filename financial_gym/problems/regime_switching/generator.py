"""Regime-switching problem generator with Bellman solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from financial_gym.base.problem import BaseProblem


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _linear_utility(x: float) -> float:
    """Risk-neutral utility: u(x) = x."""
    return x


def _exponential_utility(x: float, gamma: float) -> float:
    """Risk-averse utility: u(x) = 1 - exp(-gamma * x)."""
    return 1.0 - np.exp(-gamma * x)


# ---------------------------------------------------------------------------
# Z-grid computation
# ---------------------------------------------------------------------------

def _compute_z_grid(theta: float, sigma_z: float, kappa: float, grid_size: int) -> np.ndarray:
    """Compute problem-specific z-grid spanning ±4 std of OU stationary distribution.

    Stationary distribution is N(theta, sigma_z^2 / (2*kappa)).
    """
    stationary_std = sigma_z / np.sqrt(2 * kappa)
    z_min = theta - 4 * stationary_std
    z_max = theta + 4 * stationary_std
    return np.linspace(z_min, z_max, grid_size)


# ---------------------------------------------------------------------------
# RegimeSwitchingProblem dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeSwitchingProblem(BaseProblem):
    kappa: float          # OU mean-reversion speed
    theta: float          # OU long-run mean
    sigma_z: float        # OU signal volatility (discrete-time, per step)
    alpha: float          # Signal strength: E[X_{t+1} | Z_t] = alpha * Z_t
    sigma_x: float        # PnL noise std dev
    lam: float            # Switching cost lambda
    T: int                # Horizon
    seed: int             # Random seed
    z_path: np.ndarray    # shape (T+1,) — observable signal
    x_path: np.ndarray    # shape (T,)   — realised PnL increments
    initial_regime: int   # s_{-1}, regime before episode starts
    optimal_policy_table: np.ndarray   # shape (T, grid_size, 2)
    optimal_value: float               # E[J(s*)]

    def horizon(self) -> int:
        return self.T


# ---------------------------------------------------------------------------
# GeneratorConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
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
