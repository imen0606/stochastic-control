# Example Episodes: Good Planning vs Greedy Default

Two episodes from Opus 4 evaluation (N=500, planning zone). Same parameter regime, same kind of decision, opposite reasoning.

---

## GOOD EPISODE: Genuine Planning (seed=265)

**Parameters:** κ=0.147, α=0.4230, λ=0.1954, T=19, λ/α=0.46

**What happens:** Model starts OFF, sees three consecutive positive signals. At t=2, greedy says stay OFF (immediate PnL 0.076 < switching cost 0.195). Optimal says switch ON (persistent signal, amortise cost over remaining steps). **Model plans and matches optimal.**

### System Prompt

```
You are managing a trading strategy over T=19 steps.

RULES:
- At each step t, you choose s_t = 1 (ON) or s_t = 0 (OFF).
- If ON (s_t=1): you earn the PnL for that step. Expected PnL = 0.4230 x Z_t.
- If OFF (s_t=0): you earn nothing (PnL = 0).
- Every time you SWITCH (change from ON to OFF or OFF to ON), you pay a cost of 0.1954.
- If you stay in the same state, no switching cost.

YOUR GOAL: Maximize total profit = sum of PnL earned minus switching costs paid.

PARAMETERS:
  Signal strength: alpha = 0.4230
  Switching cost:  lambda = 0.1954
  Starting state:  s_{-1} = 0 (OFF)
  Horizon:         T = 19 steps

You will receive one signal observation Z_t at a time.
State your decision as: s_t = 0 or s_t = 1
```

### Turn 0

**User:**
```
t=0 | Z_t = +0.1187 | You are currently OFF (s_prev = 0)
```

**Opus:**
> I need to decide whether to turn ON or stay OFF at t=0.
>
> Given information:
> - Current state: OFF (s_{-1} = 0)
> - Signal: Z_0 = +0.1187
> - If I turn ON: Expected PnL = 0.4230 × 0.1187 = 0.0502
> - Switching cost if I turn ON: 0.1954
> - If I stay OFF: PnL = 0, no switching cost
>
> To decide, I need to consider whether the expected earnings from turning ON now and potentially future steps will exceed the switching cost.
>
> Since this is the first step with a positive signal, but relatively small (0.1187), and the immediate expected return (0.0502) is much less than the switching cost (0.1954), it would result in an immediate loss of about -0.145.
>
> I would need sustained positive signals to justify the switching cost. Given the weak signal strength, I'll stay OFF for now.
>
> s_0 = 0

### Turn 1

**User:**
```
t=1 | Z_t = +0.2100 | You are currently OFF (s_prev = 0)
```

**Opus:**
> I need to decide whether to turn ON or stay OFF at t=1.
>
> Given information:
> - Current state: OFF (s_0 = 0)
> - Signal: Z_1 = +0.2100
> - If I turn ON: Expected PnL = 0.4230 × 0.2100 = 0.0888
> - Switching cost if I turn ON: 0.1954
> - If I stay OFF: PnL = 0, no switching cost
>
> Now I have two consecutive positive signals (0.1187 and 0.2100). If I had turned ON at t=0, I would have earned:
> - PnL from t=0: 0.0502
> - PnL from t=1: 0.0888
> - Total: 0.1390
> - Less switching cost: 0.1390 - 0.1954 = -0.0564 (still negative)
>
> The current signal is stronger than the previous one, but still, the immediate return (0.0888) is less than the switching cost (0.1954).
>
> However, with 17 more steps remaining and two positive signals so far, I need to consider the possibility of sustained positive signals. The switching cost needs to be amortized over multiple profitable periods.
>
> Given that we're [truncated]

### Turn 2 — THE HARD DECISION

**User:**
```
t=2 | Z_t = +0.1801 | You are currently OFF (s_prev = 0)
```

**Opus (genuine planning):**
> I need to decide whether to turn ON or stay OFF at t=2.
>
> Given information:
> - Current state: OFF (s_1 = 0)
> - Signal: Z_2 = +0.1801
> - If I turn ON: Expected PnL = 0.4230 × 0.1801 = 0.0762
> - Switching cost if I turn ON: 0.1954
> - If I stay OFF: PnL = 0, no switching cost
>
> I now have three consecutive positive signals (0.1187, 0.2100, 0.1801). **If I had turned ON at t=0, I would have earned:**
> - **Total PnL: 0.0502 + 0.0888 + 0.0762 = 0.2152**
> - **Less switching cost: 0.2152 - 0.1954 = 0.0198 (slightly positive)**
>
> **The cumulative opportunity cost of staying OFF is growing.** With 16 steps remaining and a consistent pattern of positive signals, it's becoming more likely that turning ON could be profitable over the remaining horizon.
>
> Given the persistence of positive signals and the fact that the cumulative PnL would now exceed the switching cost, I'll turn ON.
>
> **s_2 = 1**

**Why this is genuine planning:** The model computes multi-step amortisation (0.0502 + 0.0888 + 0.0762 = 0.2152 > 0.1954), recognises the cumulative opportunity cost, counts remaining steps (16), and overrides the greedy decision. Greedy says "0.076 < 0.195, stay OFF." The model says "the pattern warrants switching ON."

---

## BAD EPISODE: Greedy Default (seed=190)

**Parameters:** κ=0.180, α=0.4961, λ=0.2073, T=9, λ/α=0.42

**What happens:** Model starts ON, sees weak/negative signals (-0.24, +0.08, -0.17). At t=2, optimal says switch OFF (weak signals will persist, cumulative losses exceed exit cost). Greedy says stay ON (immediate loss 0.084 < switching cost 0.207). **Model follows greedy and misses optimal.**

### System Prompt

```
You are managing a trading strategy over T=9 steps.

RULES:
- At each step t, you choose s_t = 1 (ON) or s_t = 0 (OFF).
- If ON (s_t=1): you earn the PnL for that step. Expected PnL = 0.4961 x Z_t.
- If OFF (s_t=0): you earn nothing (PnL = 0).
- Every time you SWITCH (change from ON to OFF or OFF to ON), you pay a cost of 0.2073.
- If you stay in the same state, no switching cost.

YOUR GOAL: Maximize total profit = sum of PnL earned minus switching costs paid.

PARAMETERS:
  Signal strength: alpha = 0.4961
  Switching cost:  lambda = 0.2073
  Starting state:  s_{-1} = 1 (ON)
  Horizon:         T = 9 steps

You will receive one signal observation Z_t at a time.
State your decision as: s_t = 0 or s_t = 1
```

### Turn 0

**User:**
```
t=0 | Z_t = -0.2390 | You are currently ON (s_prev = 1)
```

**Opus:**
> I need to decide whether to stay ON (s_t = 1) or switch OFF (s_t = 0).
>
> Given information:
> - Current state: ON (s_prev = 1)
> - Signal: Z_0 = -0.2390
> - Expected PnL if ON: 0.4961 × (-0.2390) = -0.1186
> - PnL if OFF: 0
> - Switching cost if I change to OFF: 0.2073
>
> Analysis:
> - If I stay ON: I lose 0.1186
> - If I switch OFF: I pay switching cost 0.2073 but avoid the negative PnL
>
> Since the negative PnL (-0.1186) is smaller in magnitude than the switching cost (0.2073), it's better to stay ON and accept the small loss rather than pay the larger switching cost.
>
> s_t = 1

### Turn 1

**User:**
```
t=1 | Z_t = +0.0758 | You are currently ON (s_prev = 1)
```

**Opus:**
> I need to decide whether to stay ON (s_t = 1) or switch OFF (s_t = 0).
>
> Given information:
> - Current state: ON (s_prev = 1)
> - Signal: Z_1 = +0.0758
> - Expected PnL if ON: 0.4961 × 0.0758 = 0.0376
> - PnL if OFF: 0
> - Switching cost if I change to OFF: 0.2073
>
> Analysis:
> - If I stay ON: I earn 0.0376
> - If I switch OFF: I pay switching cost 0.2073 and earn 0
>
> Since the signal is positive, I would earn a positive PnL by staying ON. Switching OFF would cost me 0.2073 and give up the positive earnings. This is clearly suboptimal.
>
> s_t = 1

### Turn 2 — THE HARD DECISION

**User:**
```
t=2 | Z_t = -0.1697 | You are currently ON (s_prev = 1)
```

**Opus (greedy default):**
> I need to decide whether to stay ON (s_t = 1) or switch OFF (s_t = 0).
>
> Given information:
> - Current state: ON (s_prev = 1)
> - Signal: Z_2 = -0.1697
> - Expected PnL if ON: 0.4961 × (-0.1697) = -0.0842
> - PnL if OFF: 0
> - Switching cost if I change to OFF: 0.2073
>
> Analysis:
> - If I stay ON: I lose 0.0842
> - If I switch OFF: I pay switching cost 0.2073 but avoid the negative PnL
>
> The negative PnL (-0.0842) is smaller in magnitude than the switching cost (0.2073). It's better to accept the small loss rather than pay the larger switching cost.
>
> s_t = 1

**Why this is greedy default:** The model only compares this step's loss (-0.084) to the switching cost (0.207). It never considers that the signal has been weak/negative for 2 of 3 steps, that staying ON through persistent negative signals will accumulate losses, or that paying the exit cost now might save future losses. Pure single-step cost-benefit, no forward reasoning.

---

## The Contrast

| | Good Episode | Bad Episode |
|---|---|---|
| **Hard decision** | Switch ON (override greedy OFF) | Switch OFF (override greedy ON) |
| **Greedy says** | "PnL < cost, stay OFF" | "Loss < cost, stay ON" |
| **Optimal says** | "Persistent signal, amortise cost" | "Weak signal, cut losses" |
| **Model does** | Computes cumulative PnL, switches ON | Single-step comparison, stays ON |
| **Reasoning** | Multi-step amortisation | Immediate cost-benefit only |
| **Result** | Matches optimal | Matches greedy (misses optimal) |

Both episodes are in the same parameter zone (κ~0.15-0.18, λ/α~0.42-0.46). The difference is not in the problem difficulty — it's in whether the model constructs the forward-looking reasoning scaffold.
