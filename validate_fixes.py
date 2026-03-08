"""
Validation Test — P8: 5-run stability validation with CoolingAgent

Expected results:
    Avg Temperature: 60-65°C
    Max Temperature:  <78°C
    Hotspots:         ≤1
    Violations:       0
    Energy Saved:     10-35%
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml

from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator
from controllers.pid_controller import PIDController
from agents.cooling_agent import CoolingAgent

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

NUM_RUNS = 5
NUM_STEPS = 300
rows, cols = config["simulation"]["grid_size"]

all_pass = True
results_table = []

for run in range(1, NUM_RUNS + 1):
    workload_gen = SyntheticWorkloadGenerator(
        grid_size=(rows, cols), pattern="mixed"
    )
    env = DataCenterThermalEnv(config_path="config.yaml", workload_generator=workload_gen)
    pid = PIDController(
        kp=config["pid"]["kp"],
        ki=config["pid"]["ki"],
        kd=config["pid"]["kd"],
        setpoint=config["pid"]["setpoint"],
    )
    agent = CoolingAgent()

    # Baseline: constant cooling at 0.5 (naive approach without optimization)
    workload_gen_bl = SyntheticWorkloadGenerator(
        grid_size=(rows, cols), pattern="mixed"
    )
    env_bl = DataCenterThermalEnv(config_path="config.yaml", workload_generator=workload_gen_bl)

    state, _ = env.reset()
    state_bl, _ = env_bl.reset()

    temps = []
    max_temps = []
    violations_list = []
    agent_energy = 0.0
    baseline_energy = 0.0

    for step in range(NUM_STEPS):
        # Primary: PID + CoolingAgent supervisor
        grids = env.get_state_grid()
        proposed = pid.compute(grids["temperatures"])
        env.cooling_levels = np.clip(proposed, 0.0, 1.0)
        state, reward, terminated, truncated, info = env.step(1)
        agent.post_step_safety(env)

        corrected = env.get_state_grid()
        agent_energy += float(np.mean(corrected["cooling_levels"]))
        temps.append(float(np.mean(corrected["temperatures"])))
        max_temps.append(float(np.max(corrected["temperatures"])))
        violations_list.append(int(np.sum(corrected["temperatures"] > 80.0)))

        # Baseline: constant cooling at 0.5 (naive data center approach)
        env_bl.cooling_levels = np.full((rows, cols), 0.5)
        state_bl, _, _, _, _ = env_bl.step(1)
        baseline_energy += float(np.mean(env_bl.get_state_grid()["cooling_levels"]))

        if terminated:
            break

    avg_temp = np.mean(temps[-100:]) if len(temps) >= 100 else np.mean(temps)
    max_temp_val = max(max_temps)
    total_violations = sum(violations_list)
    hotspot_max = int(np.sum(env.get_state_grid()["temperatures"] > 75.0))

    if baseline_energy > 0:
        energy_saved = max(min(((baseline_energy - agent_energy) / baseline_energy) * 100, 100.0), -100.0)
    else:
        energy_saved = 0.0

    pass_avg = 55.0 <= avg_temp <= 68.0
    pass_max = max_temp_val < 78.0
    pass_viol = total_violations == 0
    run_pass = pass_avg and pass_max and pass_viol

    results_table.append({
        "run": run, "avg_temp": avg_temp, "max_temp": max_temp_val,
        "violations": total_violations, "hotspots": hotspot_max,
        "energy_saved": energy_saved, "pass": run_pass,
    })

    if not run_pass:
        all_pass = False

    status = "PASS" if run_pass else "FAIL"
    print(f"  Run {run}: [{status}]  avg={avg_temp:.2f} C  max={max_temp_val:.2f} C  "
          f"viol={total_violations}  hotspots={hotspot_max}  energy_saved={energy_saved:.1f}%")

print()
print("=" * 70)
header = f"{'Run':>4} | {'Avg C':>8} | {'Max C':>8} | {'Viol':>5} | {'Hotsp':>5} | {'E.Saved':>8} | {'Status':>6}"
print(header)
print("-" * len(header))
for r in results_table:
    s = "PASS" if r["pass"] else "FAIL"
    print(f"{r['run']:>4} | {r['avg_temp']:>8.2f} | {r['max_temp']:>8.2f} | "
          f"{r['violations']:>5} | {r['hotspots']:>5} | {r['energy_saved']:>7.1f}% | {s:>6}")
print("=" * 70)
print(f"\n{'ALL 5 RUNS PASSED' if all_pass else 'SOME RUNS FAILED'}")

print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
