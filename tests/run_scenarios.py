"""
Section 4 — Automated Scenario Tests

Runs 6 scenarios and records metrics:
  1. Normal Operation
  2. High Ambient Temperature
  3. Workload Spike
  4. Cooling Failure (rack 1,2 disabled)
  5. Extreme Heat
  6. Long Stability Test
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator
from controllers.pid_controller import PIDController
from agents.cooling_agent import CoolingAgent


def _load_config():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _run_sim(config, *, num_steps=300, ambient=None, alpha=None,
             workload_pattern="mixed", disable_rack=None):
    """
    Run a single simulation and return metrics dict.

    Parameters
    ----------
    disable_rack : tuple or None
        (row, col) of the rack whose cooling is forced to 0 each step.
    """
    rows, cols = config["simulation"]["grid_size"]
    workload_gen = SyntheticWorkloadGenerator(
        grid_size=(rows, cols), pattern=workload_pattern,
    )
    env = DataCenterThermalEnv(
        config_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml",
        ),
        workload_generator=workload_gen,
    )
    pid = PIDController(
        kp=config["pid"]["kp"], ki=config["pid"]["ki"],
        kd=config["pid"]["kd"], setpoint=config["pid"]["setpoint"],
    )
    agent = CoolingAgent()

    if ambient is not None:
        env.ambient_temp = ambient
    if alpha is not None:
        env.heat_model.alpha = alpha

    state, _ = env.reset()

    temps_avg = []
    temps_max = []
    violations = []
    cooling_hist = []

    for step in range(num_steps):
        grids = env.get_state_grid()
        proposed = pid.compute(grids["temperatures"])
        env.cooling_levels = np.clip(proposed, 0.0, 1.0)

        # Cooling failure injection
        if disable_rack is not None:
            env.cooling_levels[disable_rack[0], disable_rack[1]] = 0.0

        state, reward, terminated, truncated, info = env.step(1)
        agent.post_step_safety(env)

        corrected = env.get_state_grid()

        # Re-disable the rack cooling after safety override to simulate
        # persistent hardware failure (safety can only help neighbours)
        if disable_rack is not None:
            env.cooling_levels[disable_rack[0], disable_rack[1]] = 0.0

        temps_avg.append(float(np.mean(corrected["temperatures"])))
        temps_max.append(float(np.max(corrected["temperatures"])))
        violations.append(int(np.sum(corrected["temperatures"] > 80.0)))
        cooling_hist.append(float(np.mean(corrected["cooling_levels"])))

        if terminated:
            break

    return {
        "avg_temp": float(np.mean(temps_avg)),
        "max_temp": float(max(temps_max)),
        "total_violations": int(sum(violations)),
        "avg_cooling": float(np.mean(cooling_hist)),
        "steps": len(temps_avg),
    }


# ---- Scenarios -------------------------------------------------------

def scenario_normal(config):
    """Scenario 1 — Normal Operation: ambient 25°C, α=0.12."""
    m = _run_sim(config, num_steps=300, ambient=25.0, alpha=0.12)
    passed = (60.0 <= m["avg_temp"] <= 65.0) and m["total_violations"] == 0
    return "Normal Operation", m, passed


def scenario_high_ambient(config):
    """Scenario 2 — High Ambient Temperature: ambient 35°C."""
    m = _run_sim(config, num_steps=300, ambient=35.0)
    passed = m["avg_temp"] < 70.0 and m["total_violations"] == 0
    return "High Ambient Temp", m, passed


def scenario_workload_spike(config):
    """Scenario 3 — Workload Spike: pattern=spikes."""
    m = _run_sim(config, num_steps=300, workload_pattern="spikes")
    passed = m["total_violations"] == 0
    return "Workload Spike", m, passed


def scenario_cooling_failure(config):
    """Scenario 4 — Cooling Failure: rack (1,2) disabled."""
    m = _run_sim(config, num_steps=300, disable_rack=(1, 2))
    # Allow the hotspot rack itself to be hot, but no violations
    passed = m["total_violations"] == 0
    return "Cooling Failure", m, passed


def scenario_extreme_heat(config):
    """Scenario 5 — Extreme Heat: α=0.20, ambient 30°C."""
    m = _run_sim(config, num_steps=300, alpha=0.20, ambient=30.0)
    passed = m["total_violations"] == 0
    return "Extreme Heat", m, passed


def scenario_long_stability(config):
    """Scenario 6 — Long Stability Test: 2000 steps."""
    m = _run_sim(config, num_steps=2000, ambient=25.0, alpha=0.12)
    passed = m["total_violations"] == 0
    return "Long Stability", m, passed


# ---- Runner ----------------------------------------------------------

ALL_SCENARIOS = [
    scenario_normal,
    scenario_high_ambient,
    scenario_workload_spike,
    scenario_cooling_failure,
    scenario_extreme_heat,
    scenario_long_stability,
]


def run_all():
    config = _load_config()
    results = []

    print("=" * 70)
    print("Automated Scenario Tests")
    print("=" * 70)

    for fn in ALL_SCENARIOS:
        name, metrics, passed = fn(config)
        results.append((name, metrics, passed))
        status = "PASS" if passed else "FAIL"
        print(
            f"  [{status}] {name:20s}  "
            f"avg={metrics['avg_temp']:.2f}°C  "
            f"max={metrics['max_temp']:.2f}°C  "
            f"viol={metrics['total_violations']}  "
            f"steps={metrics['steps']}"
        )

    print("-" * 70)
    all_pass = all(p for _, _, p in results)
    print(f"{'ALL SCENARIOS PASSED' if all_pass else 'SOME SCENARIOS FAILED'}")
    print()
    return results, all_pass


if __name__ == "__main__":
    _, success = run_all()
    sys.exit(0 if success else 1)
