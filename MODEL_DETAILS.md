# Model Details

> AI-Based Data Center Cooling Optimization — Model Architecture & Training Configuration

---

## Reinforcement Learning Agent

| Component | Detail |
|-----------|--------|
| **Algorithm** | Deep Q-Network (DQN) |
| **Neural Network** | Fully Connected MLP (Multi-Layer Perceptron) |
| **Architecture** | Input(37) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Linear(5) |
| **Optimiser** | Adam |
| **Loss Function** | MSELoss (Mean Squared Error between Q-values and TD targets) |
| **Replay Buffer** | Experience Replay (capacity: 100 000 transitions) |
| **Exploration** | Epsilon-Greedy (ε: 1.0 → 0.05, decay: 0.995 per step) |
| **Target Network** | Hard copy every 10 episodes |
| **Gradient Clipping** | Max norm 1.0 |
| **Entropy Bonus** | Weight 0.01 on action entropy to prevent policy collapse |

---

## Input State (37 dimensions)

| Index Range | Feature | Count |
|-------------|---------|-------|
| 0 – 11 | Rack temperatures (°C) | 12 |
| 12 – 23 | CPU workload levels [0, 1] | 12 |
| 24 – 35 | Cooling levels [0, 1] | 12 |
| 36 | Ambient temperature (°C) | 1 |
| **Total** | | **37** |

---

## Action Space (5 discrete actions)

| Action | Cooling Change | Description |
|--------|---------------|-------------|
| 0 | −0.10 | Decrease cooling |
| 1 | 0.00 | Maintain current level |
| 2 | +0.05 | Slight increase |
| 3 | +0.10 | Moderate increase |
| 4 | +0.20 | Strong increase |

---

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Episodes | 800 |
| Learning rate | 0.0003 |
| Discount factor (γ) | 0.99 |
| Batch size | 64 |
| Replay buffer size | 100 000 |
| Epsilon start | 1.0 |
| Epsilon end | 0.05 |
| Epsilon decay | 0.995 |
| Target update frequency | Every 10 episodes |
| Max steps per episode | 500 |
| Device | CPU (CUDA if available) |

---

## Reward Function

```
target_temp = 65

reward = -abs(temp - target_temp) - 0.5 * cooling_energy

if temp > 80:
    reward -= 100

if temp < 22:
    reward -= 5
```

| Term | Purpose |
|------|---------|
| Temperature deviation (−\|temp − 65\|) | Penalises distance from optimal 65 °C |
| Cooling energy (−0.5) | Penalises excessive cooling to save energy |
| Overheat penalty (−100) | Severe penalty when temperatures exceed 80 °C |
| Overcool penalty (−5) | Mild penalty for unrealistically low temperatures |

---

## Environment (Digital Twin)

| Parameter | Value |
|-----------|-------|
| Simulator | Gymnasium-compatible DataCenterThermalEnv |
| Grid | 3 × 4 = 12 server racks |
| Ambient temperature | 25.0 °C |
| Heat generation (α) | 0.12 |
| Cooling efficiency (β) | 0.30 |
| Heat diffusion (γ) | 0.05 |
| Ambient effect (δ) | 0.02 |
| Temperature noise | σ = 0.1 °C |
| Workload pattern | Mixed (sinusoidal + spikes + burst) |

---

## PID Baseline Controller

| Parameter | Value |
|-----------|-------|
| Kp | 0.5 |
| Ki | 0.1 |
| Kd | 0.05 |
| Setpoint | 65.0 °C |
| Output range | [0.0, 1.0] |

---

## Safety Systems

| Threshold | Temperature | Action |
|-----------|-------------|--------|
| Target | 60 °C | Desired operating point |
| Warning | 70 °C | Gradually increase cooling |
| Hotspot | 75 °C | Enforce cooling ≥ 0.50 |
| Critical | 80 °C | Force maximum cooling |
| Emergency | 85 °C | Full cooling + episode termination |
