"""Multi-turn prompt format for stateful conversation manager."""
from financial_gym.problems.regime_switching.generator import RegimeSwitchingProblem


def setup_prompt(problem: RegimeSwitchingProblem) -> str:
    """Generate Turn 0 setup message.

    Args:
        problem: RegimeSwitchingProblem instance with parameters and initial regime.

    Returns:
        Setup prompt string for the initial conversation turn.
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
    """Generate user message for a single time step.

    Args:
        t: Current time step (0-indexed).
        z_t: Signal value at time t.
        prev_regime: Previous regime s_{t-1} (or s_{-1} if t=0).

    Returns:
        Step prompt string with current observation.
    """
    regime_label = f"s_{{-1}}" if t == 0 else f"s_{{{t-1}}}"
    return f"t={t} | Z_{t} = {z_t:+.4f} | Previous regime: {regime_label} = {prev_regime}"
