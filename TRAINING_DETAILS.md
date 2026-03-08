# Training Details

> **AI-Based Data Center Cooling Optimization using Digital Twin Simulation and Safe Reinforcement Learning**

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Training episodes | 800 |
| Learning rate | 0.0003 (Adam) |
| Batch size | 64 |
| Discount factor (γ) | 0.99 |
| Epsilon start | 1.0 |
| Epsilon end | 0.05 |
| Epsilon decay | 0.995 per step |
| Replay buffer size | 100 000 |
| Target network update | Every 10 episodes (hard copy) |
| Entropy weight | 0.01 |
| Gradient clipping | Max norm 1.0 |
| Max steps per episode | 500 |
| Device | CPU |

---

## Network Architecture

```
Input (37)  ─►  Linear(37, 256)  ─►  ReLU
            ─►  Linear(256, 256) ─►  ReLU
            ─►  Linear(256, 128) ─►  ReLU
            ─►  Linear(128, 5)   ─►  Q-values
```

**State vector (37 dimensions):**
- 12 rack temperatures (normalised)
- 12 CPU workload levels
- 12 cooling levels
- 1 ambient temperature

**Action space (5 discrete):**

| Action | Cooling Change | Description |
|--------|---------------|-------------|
| 0 | −0.10 | Decrease cooling |
| 1 | 0.00 | Maintain current |
| 2 | +0.05 | Slight increase |
| 3 | +0.10 | Moderate increase |
| 4 | +0.20 | Strong increase |

---

## Reward Function

```
reward = (
    −0.6 × energy_cost
    − 15 × max(0, avg_temperature − 65)
    − 200 × violations
    + max(0, 0.35 − energy_cost) × 10      # energy-saving bonus
    − |current_temp − previous_temp| × 2    # stability penalty
)
```

### Design Rationale

| Component | Weight | Purpose |
|-----------|--------|---------|
| Energy cost | −0.6 | Penalises excessive cooling to reduce energy waste |
| Temperature excess | −15 | Strong penalty when avg temperature exceeds 65 °C |
| Violations | −200 | Severe penalty for racks above 80 °C (safety-critical) |
| Energy bonus | +10 | Bonus for keeping avg cooling below 0.35 |
| Stability penalty | −2 | Discourages rapid temperature swings |

---

## Epsilon Decay Schedule

The exploration rate follows exponential decay:

$$\varepsilon_{t+1} = \max(0.05,\; \varepsilon_t \times 0.995)$$

| Step | ε (approx.) |
|------|-------------|
| 0 | 1.000 |
| 100 | 0.606 |
| 200 | 0.367 |
| 500 | 0.082 |
| 600 | 0.050 (floored) |

---

## Training Environment

| Parameter | Value |
|-----------|-------|
| Simulator | Gymnasium-compatible Digital Twin |
| Grid | 3 × 4 (12 server racks) |
| Ambient temperature | 25.0 °C |
| Heat generation (α) | 0.12 |
| Cooling efficiency (β) | 0.30 |
| Heat diffusion (γ) | 0.05 |
| Ambient effect (δ) | 0.02 |
| Workload pattern | Mixed (sinusoidal + spikes + burst) |
| Base / Peak load | 0.3 / 0.9 |

---

## Safety Mechanisms During Training

The RL agent trains with safety constraints active:

1. **Cooling floor**: If avg temperature > 65 °C, minimum cooling = 0.20–0.40
2. **Proactive gradient cooling**: If temperature rising > 2 °C/step, boost cooling
3. **Per-rack escalation**: 65–70 °C moderate, 70–75 °C high, >75 °C strong, >80 °C max
4. **Post-step override**: Emergency override corrects any remaining violations

These mechanisms ensure the agent cannot cause dangerous states even during exploration.

---

## Expected Training Outcomes

| Metric | Target Range |
|--------|-------------|
| Average temperature | 63–65 °C |
| Maximum temperature | < 75 °C |
| Hotspots (>75 °C) | 0 |
| Violations (>80 °C) | 0 |
| Energy saved vs PID | 10–30 % |
| Episode length | 500 steps (no early termination) |

---

## Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/dqn_episode_50.pth` | Early training snapshot |
| `checkpoints/dqn_episode_100.pth` | Progress checkpoint |
| `checkpoints/dqn_final.pth` | Final trained model |

---

## Training Logs

| File | Contents |
|------|----------|
| `logs/episode_logs.csv` | Per-episode: reward, avg/max temp, cooling, violations, epsilon, loss |
| `logs/training_logs.csv` | Per-step: reward, temperature, cooling, action, loss |
| `logs/training_progress_ep*.png` | Progress plots at checkpoint intervals |

---

## Reproducibility

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Train with default 800 episodes
python train_model.py --episodes 800

# Resume from checkpoint
python train_model.py --episodes 800 --resume checkpoints/dqn_episode_100.pth
```
