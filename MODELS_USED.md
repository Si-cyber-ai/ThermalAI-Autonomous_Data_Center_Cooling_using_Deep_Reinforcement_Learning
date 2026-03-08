# Models Used

> **AI-Based Data Center Cooling Optimization using Digital Twin Simulation and Safe Reinforcement Learning**

---

## 1. Reinforcement Learning Controller — Deep Q-Network (DQN)

| Property | Value |
|----------|-------|
| **Algorithm** | Deep Q-Network (DQN) with experience replay and target network |
| **Network architecture** | MLP: 37 → 256 → 256 → 128 → 5 (ReLU activations) |
| **State space** | 37-dimensional vector: 12 rack temperatures + 12 CPU workloads + 12 cooling levels + 1 ambient temperature |
| **Action space** | 5 discrete actions: Decrease (−0.1), Maintain (0.0), Slight increase (+0.05), Moderate increase (+0.1), Strong increase (+0.2) |
| **Learning rate** | 0.0003 (Adam optimiser) |
| **Discount factor (γ)** | 0.99 |
| **Exploration** | ε-greedy, decaying from 1.0 → 0.05 with decay rate 0.995 |
| **Batch size** | 64 |
| **Replay buffer** | 100 000 transitions |
| **Target network update** | Every 10 episodes (hard copy) |
| **Entropy bonus** | 0.01 weight on action entropy to prevent policy collapse |
| **Gradient clipping** | Max norm 1.0 |

### Reward Function

```
reward = (
    −0.6 × energy_cost
    − 15 × max(0, avg_temperature − 65)
    − 200 × violations
    + max(0, 0.35 − energy_cost) × 10          # energy-saving bonus
    − |current_temp − previous_temp| × 2        # stability penalty
)
```

Where:
- `energy_cost` = mean(cooling_levels)
- `avg_temperature` = mean(rack_temperatures)
- `violations` = number of racks above 80 °C

---

## 2. PID Controller (Baseline)

| Property | Value |
|----------|-------|
| **Type** | Proportional-Integral-Derivative controller |
| **Kp** | 0.5 |
| **Ki** | 0.1 |
| **Kd** | 0.05 |
| **Setpoint** | 65.0 °C |
| **Output range** | [0.0, 1.0] (cooling level) |
| **Anti-windup** | Integral clamped to [−10, +10] |

### Control Equation

$$
u(t) = K_p \cdot e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d \frac{de(t)}{dt}
$$

Where $e(t) = T_{current} - T_{setpoint}$.

---

## 3. Digital Twin — Heat Transfer Model

| Property | Value |
|----------|-------|
| **Grid size** | 3 × 4 = 12 server racks |
| **Heat generation (α)** | 0.12 — CPU workload → heat |
| **Cooling efficiency (β)** | 0.30 — cooling → heat removal |
| **Heat diffusion (γ)** | 0.05 — rack-to-rack conduction |
| **Ambient effect (δ)** | 0.02 — ambient temperature influence |
| **Noise** | Gaussian, σ = 0.1 °C |
| **Neighbour kernel** | 3×3 convolution with airflow-weighted coefficients |

### Thermal Dynamics Equation

$$
T_{i,j}(t+1) = T_{i,j}(t) + \Delta t \left[
    \alpha \cdot W_{i,j}
    - \beta \cdot C_{i,j}
    + \gamma \cdot H_{neighbours}
    + \delta \cdot (T_{ambient} - T_{i,j})
\right] + \mathcal{N}(0, \sigma^2)
$$

Where:
- $W_{i,j}$ = CPU workload at rack $(i,j)$ ∈ [0, 1]
- $C_{i,j}$ = cooling level at rack $(i,j)$ ∈ [0, 1]
- $H_{neighbours}$ = convolution of neighbour temperatures
- $T_{ambient}$ = data centre ambient temperature (25 °C default)

---

## 4. Supervisory AI Cooling Agent

| Property | Value |
|----------|-------|
| **Role** | High-level supervisor above RL and PID controllers |
| **Input** | Temperature grid from Digital Twin |
| **Output** | Control strategy selection + safety-filtered cooling levels |

### Decision Logic

| Max Temperature | Risk Level | Strategy |
|----------------|------------|----------|
| ≤ 65 °C | Normal | RL Agent takes control |
| 65–70 °C | Moderate Risk | Switch to PID controller |
| 70–75 °C | High Risk | Fixed strong cooling (0.7–0.85) |
| > 80 °C | Emergency | Maximum cooling (1.0) on all racks |

### Pipeline

```
Digital Twin → CoolingAgent.observe()
            → evaluate_risk()
            → choose_strategy()
            → Controller (RL / PID / Direct)
            → SafetyFilter.apply()
            → Environment
```

---

## 5. Safety Filter

| Threshold | Temperature | Action |
|-----------|-------------|--------|
| Target | 60 °C | Desired operating point |
| Warning | 70 °C | Gradually increase cooling (0.35–0.50) |
| Hotspot | 75 °C | Enforce cooling ≥ 0.50 |
| Critical | 80 °C | Force maximum cooling (1.0) |
| Emergency | 85 °C | Full shutdown-level cooling + episode termination |

The safety filter is applied **after every controller action** and cannot be bypassed by the RL agent.

---

## 6. Workload Generator

| Mode | Description |
|------|-------------|
| Sinusoidal | Smooth periodic load variation |
| Spikes | Random sharp load bursts |
| Burst | Sustained high-load periods |
| Mixed | Combination of all patterns (default) |

Workload range: base load 0.3 → peak load 0.9.
