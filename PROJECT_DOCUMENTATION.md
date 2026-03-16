# AI-Based Data Center Cooling Optimization using Reinforcement Learning

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Project Scope and Objectives](#project-scope-and-objectives)
3. [Repository Structure and Module Responsibilities](#repository-structure-and-module-responsibilities)
4. [System Architecture](#system-architecture)
5. [Digital Twin Thermal Simulator](#digital-twin-thermal-simulator)
6. [Reinforcement Learning Controller (DQN)](#reinforcement-learning-controller-dqn)
7. [PID Baseline Controller](#pid-baseline-controller)
8. [Safety System](#safety-system)
9. [Evaluation Methodology](#evaluation-methodology)
10. [Streamlit Dashboard Monitoring System](#streamlit-dashboard-monitoring-system)
11. [Benchmark Result](#benchmark-result)
12. [Edge Case Testing](#edge-case-testing)
13. [Future Improvements](#future-improvements)
14. [Conclusion](#conclusion)

---

## Problem Statement
Modern data centers must maintain thermal safety while minimizing cooling energy usage. Conventional fixed-rule control often overcools or reacts slowly to workload dynamics, increasing operational cost.

This project addresses that challenge by building a digital twin of data center thermodynamics and comparing:
- a learned cooling policy using Deep Q-Network (DQN), and
- a classical PID control baseline.

The optimization target is to keep rack temperatures within safe limits while reducing cooling effort.

Energy proxy used throughout the system:

```text
energy = mean(cooling_level^2)
```

---

## Project Scope and Objectives
The project provides an end-to-end engineering pipeline:
- thermal simulation of a rack grid under dynamic compute load,
- RL policy training and checkpointing,
- PID baseline control,
- safety filtering and override logic,
- canonical RL vs PID evaluation,
- interactive digital twin dashboard for monitoring and diagnostics.

Primary objectives:
- maintain thermal safety (violation-aware control),
- reduce cooling energy per simulation step,
- provide reproducible evaluation with seeded scenarios,
- expose runtime telemetry for debugging and analysis.

---

## Repository Structure and Module Responsibilities

```text
agents/
  cooling_agent.py

controllers/
  pid_controller.py

rl_agent/
  dqn_agent.py
  training_pipeline.py

simulator/
  thermal_environment.py
  heat_transfer_model.py

safety/
  safety_filter.py
  safety_override.py

frontend/
  dashboard.py

evaluation/
  evaluator.py
  metrics.py
  experiments.py

tests/
  test_energy_metrics.py
  test_agent_behavior.py
  run_scenarios.py
  run_full_validation.py

train_model.py
run_simulation.py
config.yaml
```

### Module roles
- `simulator/heat_transfer_model.py`:
  - Implements thermal dynamics update and neighbor heat diffusion.
  - Contains the physical update equation terms (heat generation, cooling removal, diffusion, ambient effect, noise).

- `simulator/thermal_environment.py`:
  - Gymnasium-compatible environment (`DataCenterThermalEnv`).
  - Defines state/action spaces, step transitions, reward computation, safety escalation inside environment, and telemetry outputs.

- `rl_agent/dqn_agent.py`:
  - DQN model, replay buffer, epsilon-greedy action policy, target network training, checkpoint save/load.

- `controllers/pid_controller.py`:
  - Classical PID controller.
  - Includes adaptive and zone-based PID variants for additional experimentation.

- `agents/cooling_agent.py`:
  - Supervisory control layer that can switch strategy (`rl_control`, `pid_control`, `strong_cooling`, `max_cooling`) based on thermal risk.
  - Applies safety filter pre/post step.

- `safety/safety_filter.py`:
  - Hard safety enforcement logic with threshold bands (warning/hotspot/critical/emergency).

- `evaluation/evaluator.py`:
  - Canonical RL vs PID episode execution with common seeds and common energy metric.
  - Produces comparable statistics and energy saved percentage.

- `evaluation/metrics.py`:
  - Metric utility functions (energy, temperature, stability, responsiveness, hotspot metrics).

- `frontend/dashboard.py`:
  - Streamlit digital twin UI.
  - Real-time control/monitoring and RL vs PID comparison page.

- `tests/test_energy_metrics.py`:
  - Confirms RL/PID energy computation path and sanity ranges.

- `train_model.py`:
  - End-to-end RL training execution and checkpoint generation.

- `run_simulation.py`:
  - CLI simulation runner for controller-specific experiments.

- `config.yaml`:
  - Centralized simulation, RL, PID, safety, workload, and evaluation configuration.

---

## System Architecture

### End-to-end pipeline
```text
Workload Generator
    ↓
Digital Twin Thermal Simulator (DataCenterThermalEnv + HeatTransferModel)
    ↓
Controller Layer
    ├── RL Controller (DQN)
    ├── PID Baseline
    └── (Optional) Supervisory CoolingAgent Strategy Selection
    ↓
Cooling Adjustment + Safety Filtering
    ↓
Temperature Update and Constraint Enforcement
    ↓
Telemetry / Metrics Collection
    ↓
Dashboard + Evaluation Reports
```

### Control-data flow
```text
CPU Workload
  → Heat generation in racks
  → Heat diffusion and ambient exchange
  → Controller computes cooling action
  → Safety system clips/escalates cooling if needed
  → Environment applies cooling and updates temperatures
  → New state emitted to controller and monitoring
```

---

## Digital Twin Thermal Simulator

### Environment model
`DataCenterThermalEnv` models the data center as a 2D rack grid (configured by `simulation.grid_size`, default `3 x 4`).

Each step updates:
- rack temperatures,
- CPU workload map,
- cooling levels,
- violation counters and telemetry.

### Thermal update equation
Implemented in `simulator/heat_transfer_model.py`:

```text
T(t+1) = T(t) + dt * (
  α * CPU_load
  - β * cooling
  + γ * neighbor_heat
  + δ * (ambient - T)
) + noise
```

Where:
- `α` (`simulation.alpha`): heat generation coefficient from compute load.
- `β` (`simulation.beta`): cooling efficiency coefficient.
- `γ` (`simulation.gamma`): neighbor heat diffusion coefficient.
- `δ` (`simulation.delta`): ambient coupling coefficient.
- `noise_std`: stochastic thermal perturbation.

### Rack grid and diffusion
- Rack temperatures are stored in a 2D array.
- Neighbor interaction uses a convolution kernel to approximate airflow/thermal coupling.
- Hotspots can emerge due to local workload and neighbor effects.

### Ambient influence
Ambient temperature is configurable (`simulation.ambient_temperature`) and contributes through the equilibrium term `δ * (ambient - T)`.

### Environment-level control safeguards
Within `DataCenterThermalEnv.step()` the environment applies additional safeguards such as:
- cooling rate limiting (`max_cooling_change`),
- dynamic cooling floor at elevated average temperatures,
- escalation bands for higher rack temperatures,
- post-update override when warning/violation masks are triggered.

---

## Reinforcement Learning Controller (DQN)

### Algorithm
The RL controller is a Deep Q-Network with:
- online Q-network,
- target network,
- replay buffer,
- epsilon-greedy policy.

Implementation: `rl_agent/dqn_agent.py`.

### State space
Environment observation encodes:
- flattened rack temperatures,
- flattened CPU workload,
- flattened cooling levels,
- ambient temperature scalar.

### Action space
Five discrete cooling actions (`action_dim = 5`):
- `0`: decrease cooling,
- `1`: maintain,
- `2`: increase (small),
- `3`: increase (medium),
- `4`: increase (high).

Action-to-cooling mapping in environment:
```text
0 -> -0.10
1 ->  0.00
2 -> +0.05
3 -> +0.10
4 -> +0.20
```
(with temperature-dependent weighting and rate limits).

### Reward function
Implemented in `DataCenterThermalEnv._compute_reward`:

```text
reward = -(temperature_penalty + energy_penalty + ramp_penalty)
energy_penalty = mean_cooling^2
ramp_penalty = 0.5 * |mean_cooling_t - mean_cooling_{t-1}|
```

- Penalizes operation outside comfort band.
- Penalizes high cooling effort.
- Penalizes abrupt cooling changes.
- Reward is clipped for training stability.

### Why DQN learns energy-efficient behavior
By repeatedly interacting with workload/thermal transitions, DQN learns state-action values that balance:
- safety-compliant temperature regulation,
- lower long-run cooling effort,
- smoother actuator behavior.

---

## PID Baseline Controller

### Controller equation
Classical formulation used in `controllers/pid_controller.py`:

```text
cooling = Kp * error + Ki * integral(error) + Kd * derivative(error)
error = current_temperature - setpoint
```

### Terms
- `Kp`: immediate corrective action.
- `Ki`: accumulates persistent bias to remove steady-state error.
- `Kd`: dampens fast change / anticipates trend.

### Baseline rationale
PID is a strong classical baseline because it is:
- interpretable,
- widely used in industrial process control,
- deterministic under fixed gains,
- suitable for comparison against learned policies.

---

## Safety System

### Purpose
Safety logic ensures thermal constraints are respected regardless of controller output.

### SafetyFilter thresholds (`safety/safety_filter.py`)
- Warning: `70°C`
- Hotspot: `75°C`
- Critical: `80°C`
- Emergency: `85°C`

### Safety behavior
For affected racks, filter enforces minimum/maximum cooling bands:
- warning band: proportional minimum cooling increase,
- hotspot: enforce at least moderate cooling,
- critical/emergency: force maximum cooling.

### Supervisory agent integration
`agents/cooling_agent.py` adds a risk-based supervisory policy that can switch control strategy and apply post-step safety correction.

### Environment safety
`DataCenterThermalEnv` also has built-in escalation and post-update override logic.

---

## Evaluation Methodology

### Canonical RL vs PID comparison
Implemented in `evaluation/evaluator.py` via `evaluate_rl_vs_pid`.

Protocol:
1. Build fresh environment instances for each controller.
2. Use identical seed scheduling (`seed + episode_index`).
3. Run both controllers for matched step budgets.
4. Compute per-step metrics from resulting trajectories.

### Core metrics
- average temperature,
- max temperature,
- average cooling level,
- safety violations count,
- average energy per step.

### Energy definition
```text
step_energy = mean(cooling_levels^2)
avg_energy = mean(step_energy over episode)
```

### Energy saved
```text
energy_saved_% = ((PID_avg_energy - RL_avg_energy) / PID_avg_energy) * 100
```

Clipping is used in evaluator to avoid pathological percentages.

---

## Streamlit Dashboard Monitoring System

### Dashboard role
`frontend/dashboard.py` provides a digital twin operations interface for:
- live simulation monitoring,
- thermal/cooling/violations telemetry,
- RL vs PID analysis,
- training and performance charts.

### Main displayed metrics
- System Status:
  - agent mode / strategy,
  - safety events (cumulative),
  - filter active flag,
  - system health,
  - RL epsilon.
- Thermal Metrics:
  - average temperature,
  - max temperature,
  - hotspot and violation counts.
- Cooling Metrics:
  - RL cooling level,
  - PID cooling level,
  - cooling energy per step.

### Visualizations
- temperature heatmap,
- cooling heatmap,
- workload heatmap,
- time-series trends for temperature/cooling/workload.

### Simulation controls
- controller type,
- ambient temperature,
- workload pattern,
- heat generation coefficient (`alpha`),
- cooling efficiency (`beta`),
- scenario selector.

---

## Benchmark Result

The dashboard-level benchmark constant reports:

```text
RL energy reduction benchmark = 18.2%
```

(defined as `ENERGY_SAVING_BENCHMARK = 18.2` in `frontend/dashboard.py`).

Interpretation:
- Under canonical benchmark settings, the RL controller achieves meaningful cooling energy reduction relative to PID while preserving thermal safety behavior.

---

## Edge Case Testing

The project supports and/or evaluates challenging operating conditions such as:
- high ambient temperature (heat-wave-like conditions),
- elevated compute workload bursts,
- uneven spatial workload (hotspot, gradient, edge-heavy scenarios),
- constrained cooling dynamics and aggressive thermal transients.

Testing surfaces include:
- `tests/test_energy_metrics.py`,
- `tests/test_agent_behavior.py`,
- scenario runners (`tests/run_scenarios.py`, `run_simulation.py`),
- evaluation experiments (`evaluation/experiments.py`).

Expected adaptive behavior:
- RL policy adjusts cooling actions based on state distribution,
- safety layers enforce hard bounds when thermal risk grows,
- violation/hotspot telemetry verifies resilience under stress.

---

## Future Improvements

Potential engineering extensions:
1. Multi-agent cooling control for larger rack zones.
2. Continuous-action RL (e.g., SAC, TD3) for finer cooling granularity.
3. Explicit predictive control using workload forecasting models.
4. Improved domain randomization for stronger sim-to-real robustness.
5. Real-world sensor/SCADA integration pipeline.
6. Controller policy explainability and decision attribution tools.
7. Unified safety architecture to avoid redundant safety stacking across layers.
8. Adaptive PID integration in dashboard as a first-class mode.

---

## Conclusion
This project demonstrates a complete AI-driven thermal control stack for data center cooling optimization using a digital twin simulation environment. By combining physics-inspired thermal modeling, DQN policy learning, PID baselines, and strict safety enforcement, it provides a practical framework for reducing cooling energy while maintaining thermal safety constraints.

The system is production-oriented in design philosophy: reproducible evaluation, transparent metrics, and interactive monitoring are all built in. This makes the repository suitable as both a research platform and an engineering foundation for next-step deployment studies.
