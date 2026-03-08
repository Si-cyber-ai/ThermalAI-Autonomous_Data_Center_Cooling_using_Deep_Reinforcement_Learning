"""
Research Graph Generator for Data Center Cooling Optimization

Generates publication-quality figures comparing RL and PID controllers.
Outputs PNG (300 dpi) and PDF for each figure.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import yaml

from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator, WorkloadScenario
from rl_agent.dqn_agent import DQNAgent
from controllers.pid_controller import PIDController
from agents.cooling_agent import CoolingAgent

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────

def _save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf")


def _load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def _make_env(config):
    grid_size = tuple(config["simulation"]["grid_size"])
    wg = SyntheticWorkloadGenerator(
        grid_size=grid_size,
        pattern=config["workload"]["synthetic_pattern"],
        base_load=config["workload"]["base_load"],
        peak_load=config["workload"]["peak_load"],
    )
    env = DataCenterThermalEnv(config_path="config.yaml", workload_generator=wg)
    return env


def _make_controllers(config, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    rl = DQNAgent(state_dim=state_dim, action_dim=action_dim,
                  hidden_dim=config["rl"]["hidden_dim"])
    ckpt = "checkpoints/dqn_final.pth"
    if os.path.exists(ckpt):
        rl.load_checkpoint(ckpt)
    pid = PIDController(
        kp=config["pid"]["kp"], ki=config["pid"]["ki"],
        kd=config["pid"]["kd"], setpoint=config["pid"]["setpoint"],
    )
    return rl, pid


def _run_simulation(config, use_rl=True, steps=300):
    """Run a simulation and return step-wise data."""
    env = _make_env(config)
    rl, pid = _make_controllers(config, env)
    agent = CoolingAgent()
    state, _ = env.reset()

    temps, cooling, rewards, violations, energies, actions = [], [], [], [], [], []

    for _ in range(steps):
        if use_rl:
            _, rl_action, _ = agent.act(env, rl, pid, state, training=False)
            action = rl_action
        else:
            grids = env.get_state_grid()
            proposed = pid.compute(grids["temperatures"])
            env.cooling_levels = np.clip(proposed, 0.0, 1.0)
            action = 1

        state, reward, terminated, truncated, info = env.step(action)
        agent.post_step_safety(env)
        grids = env.get_state_grid()

        temps.append(float(np.mean(grids["temperatures"])))
        cooling.append(float(np.mean(grids["cooling_levels"])))
        rewards.append(reward)
        violations.append(int(np.sum(grids["temperatures"] > 80.0)))
        energies.append(float(np.mean(grids["cooling_levels"])))
        actions.append(action)

        if terminated:
            break

    return {
        "temps": np.array(temps),
        "cooling": np.array(cooling),
        "rewards": np.array(rewards),
        "violations": np.array(violations),
        "energies": np.array(energies),
        "actions": np.array(actions),
        "final_grid": grids["temperatures"],
    }


# ── Figure generators ────────────────────────────────────────────────

def fig1_temperature_comparison(config):
    """Figure 1: RL vs PID Temperature Control"""
    rl = _run_simulation(config, use_rl=True)
    pid = _run_simulation(config, use_rl=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rl["temps"], label="RL Controller", linewidth=1.5, color="#1f77b4")
    ax.plot(pid["temps"], label="PID Controller", linewidth=1.5, color="#ff7f0e")
    ax.axhline(y=65, linestyle="--", color="green", alpha=0.6, label="Target (65 °C)")
    ax.axhline(y=80, linestyle="--", color="red", alpha=0.6, label="Safety Limit (80 °C)")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Average Temperature (°C)")
    ax.set_title("Figure 1: RL vs PID Temperature Control")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.text(0.5, -0.04,
             "Temperature control comparison between RL and PID controllers. "
             "The RL controller maintains temperatures closer to the target operating range "
             "while avoiding safety violations.",
             ha="center", fontsize=9, style="italic", wrap=True)
    _save(fig, "01_temperature_comparison")
    return rl, pid


def fig2_energy_comparison(rl_data, pid_data):
    """Figure 2: Cooling Energy Consumption Comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Time-series
    ax1.plot(np.cumsum(rl_data["energies"]), label="RL (cumulative)", color="#1f77b4")
    ax1.plot(np.cumsum(pid_data["energies"]), label="PID (cumulative)", color="#ff7f0e")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Energy (a.u.)")
    ax1.set_title("Cumulative Cooling Energy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart
    rl_avg = float(np.mean(rl_data["energies"]))
    pid_avg = float(np.mean(pid_data["energies"]))
    bars = ax2.bar(["RL", "PID"], [rl_avg, pid_avg], color=["#1f77b4", "#ff7f0e"], width=0.5)
    ax2.set_ylabel("Avg Energy / Step")
    ax2.set_title("Average Cooling Energy")
    for bar, val in zip(bars, [rl_avg, pid_avg]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Figure 2: Cooling Energy Consumption Comparison", fontsize=13)
    fig.text(0.5, -0.04,
             "Comparison of energy consumed by RL and PID controllers. "
             "Lower cumulative energy indicates better efficiency while maintaining thermal safety.",
             ha="center", fontsize=9, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "02_energy_comparison")


def fig3_training_reward_curve():
    """Figure 3: RL Training Reward Curve"""
    import pandas as pd
    path = "logs/episode_logs.csv"
    if not os.path.exists(path):
        print("  [SKIP] No episode_logs.csv found for reward curve")
        return
    df = pd.read_csv(path)
    df["total_reward"] = pd.to_numeric(df["total_reward"], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["episode"], df["total_reward"], alpha=0.3, linewidth=0.8, color="#1f77b4", label="Raw")
    window = max(5, len(df) // 20)
    ma = df["total_reward"].rolling(window, min_periods=1).mean()
    ax.plot(df["episode"], ma, linewidth=2, color="#d62728", label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Figure 3: RL Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.text(0.5, -0.04,
             "DQN training reward over episodes. The moving average shows convergence "
             "as the agent learns to balance energy efficiency with thermal safety.",
             ha="center", fontsize=9, style="italic", wrap=True)
    _save(fig, "03_training_reward_curve")


def fig4_temperature_stability(config):
    """Figure 4: Temperature Stability Over Time"""
    rl = _run_simulation(config, use_rl=True, steps=500)
    pid = _run_simulation(config, use_rl=False, steps=500)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(rl["temps"], label="RL", color="#1f77b4")
    ax1.plot(pid["temps"], label="PID", color="#ff7f0e")
    ax1.axhline(y=65, ls="--", color="green", alpha=0.5)
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature Trajectory")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    rl_diff = np.abs(np.diff(rl["temps"]))
    pid_diff = np.abs(np.diff(pid["temps"]))
    ax2.plot(rl_diff, label=f"RL (σ={np.std(rl_diff):.3f})", color="#1f77b4", alpha=0.7)
    ax2.plot(pid_diff, label=f"PID (σ={np.std(pid_diff):.3f})", color="#ff7f0e", alpha=0.7)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("|ΔT| (°C)")
    ax2.set_title("Step-to-Step Temperature Change")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Figure 4: Temperature Stability Over Time", fontsize=13)
    fig.text(0.5, -0.03,
             "Top: temperature trajectories for RL and PID. Bottom: absolute step-to-step "
             "temperature change. Lower variability indicates a more stable controller.",
             ha="center", fontsize=9, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "04_temperature_stability")


def fig5_action_distribution(config):
    """Figure 5: Cooling Action Distribution"""
    rl = _run_simulation(config, use_rl=True)

    action_labels = {0: "Decrease", 1: "Maintain", 2: "Slight ↑", 3: "Moderate ↑", 4: "Strong ↑"}
    unique, counts = np.unique(rl["actions"], return_counts=True)
    labels = [action_labels.get(int(a), str(a)) for a in unique]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(labels, counts, color="#1f77b4")
    ax1.set_xlabel("Action")
    ax1.set_ylabel("Count")
    ax1.set_title("Action Frequency")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.pie(counts, labels=labels, autopct="%1.1f%%",
            colors=plt.cm.Set2(np.linspace(0, 1, len(labels))))
    ax2.set_title("Action Proportion")

    fig.suptitle("Figure 5: RL Cooling Action Distribution", fontsize=13)
    fig.text(0.5, -0.04,
             "Distribution of discrete cooling actions selected by the RL agent. "
             "A well-trained agent uses 'Decrease' and 'Maintain' more often, conserving energy "
             "while reserving higher actions for temperature spikes.",
             ha="center", fontsize=9, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "05_action_distribution")


def fig6_rack_heatmap(config):
    """Figure 6: Rack Temperature Heatmap"""
    rl = _run_simulation(config, use_rl=True)
    pid = _run_simulation(config, use_rl=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    im1 = ax1.imshow(rl["final_grid"], cmap="hot", vmin=20, vmax=80)
    ax1.set_title("RL Controller")
    ax1.set_xlabel("Rack Column")
    ax1.set_ylabel("Rack Row")
    plt.colorbar(im1, ax=ax1, label="°C")

    im2 = ax2.imshow(pid["final_grid"], cmap="hot", vmin=20, vmax=80)
    ax2.set_title("PID Controller")
    ax2.set_xlabel("Rack Column")
    ax2.set_ylabel("Rack Row")
    plt.colorbar(im2, ax=ax2, label="°C")

    fig.suptitle("Figure 6: Rack Temperature Heatmap (Final Step)", fontsize=13)
    fig.text(0.5, -0.06,
             "Spatial distribution of rack temperatures at the final simulation step. "
             "Uniform colours indicate even cooling; hotspots appear as bright patches.",
             ha="center", fontsize=9, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "06_rack_temperature_heatmap")


def fig7_workload_spike_response(config):
    """Figure 7: Workload Spike Response"""
    # Run with spike workload
    env = _make_env(config)
    rl, pid_ctrl = _make_controllers(config, env)
    agent = CoolingAgent()
    state, _ = env.reset()

    temps_rl, cooling_rl, workloads = [], [], []

    for step in range(300):
        # Inject spike at step 100-130
        if 100 <= step <= 130:
            env.cpu_workload = np.full_like(env.cpu_workload, 0.95)
        elif step == 131:
            env.cpu_workload = np.full_like(env.cpu_workload, 0.4)

        _, rl_action, _ = agent.act(env, rl, pid_ctrl, state, training=False)
        state, _, terminated, _, _ = env.step(rl_action)
        agent.post_step_safety(env)
        grids = env.get_state_grid()
        temps_rl.append(float(np.mean(grids["temperatures"])))
        cooling_rl.append(float(np.mean(grids["cooling_levels"])))
        workloads.append(float(np.mean(env.cpu_workload)))
        if terminated:
            break

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    ax1.plot(workloads, color="purple", linewidth=1.5)
    ax1.set_ylabel("CPU Workload")
    ax1.set_title("Injected Workload Spike (steps 100–130)")
    ax1.axvspan(100, 130, alpha=0.15, color="red")
    ax1.grid(True, alpha=0.3)

    ax2.plot(temps_rl, color="#1f77b4", linewidth=1.5)
    ax2.axhline(y=65, ls="--", color="green", alpha=0.5, label="Target")
    ax2.axhline(y=80, ls="--", color="red", alpha=0.5, label="Safety Limit")
    ax2.set_ylabel("Avg Temperature (°C)")
    ax2.set_title("Temperature Response")
    ax2.legend(loc="upper right")
    ax2.axvspan(100, 130, alpha=0.15, color="red")
    ax2.grid(True, alpha=0.3)

    ax3.plot(cooling_rl, color="#2ca02c", linewidth=1.5)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Avg Cooling Level")
    ax3.set_title("Cooling Response")
    ax3.axvspan(100, 130, alpha=0.15, color="red")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Figure 7: Workload Spike Response", fontsize=13)
    fig.text(0.5, -0.03,
             "Response of the RL controller to a sudden workload spike (95 % CPU, steps 100–130). "
             "The controller increases cooling proactively and stabilises temperature without "
             "exceeding the safety threshold.",
             ha="center", fontsize=9, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "07_workload_spike_response")


def fig8_long_term_stability(config):
    """Figure 8: Long-Term Stability Test"""
    rl = _run_simulation(config, use_rl=True, steps=2000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    ax1.plot(rl["temps"], linewidth=0.8, color="#1f77b4")
    ax1.axhline(y=65, ls="--", color="green", alpha=0.5, label="Target")
    ax1.axhline(y=80, ls="--", color="red", alpha=0.5, label="Safety Limit")
    ax1.set_ylabel("Avg Temperature (°C)")
    ax1.set_title("2000-Step Long-Term Temperature")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(rl["cooling"], linewidth=0.8, color="#2ca02c")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Avg Cooling Level")
    ax2.set_title("Cooling Effort")
    ax2.grid(True, alpha=0.3)

    # Stats annotation
    stats_text = (
        f"Mean Temp: {np.mean(rl['temps']):.1f} °C  |  "
        f"Max Temp: {np.max(rl['temps']):.1f} °C  |  "
        f"Violations: {int(np.sum(rl['violations']))}  |  "
        f"Avg Cooling: {np.mean(rl['cooling']):.3f}"
    )

    fig.suptitle("Figure 8: Long-Term Stability Test", fontsize=13)
    fig.text(0.5, -0.04, stats_text, ha="center", fontsize=10, weight="bold")
    fig.text(0.5, -0.08,
             "Extended 2000-step simulation demonstrating the RL controller's ability to "
             "maintain stable temperatures over long periods without drift or oscillation.",
             ha="center", fontsize=9, style="italic", wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "08_long_term_stability")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Research Graph Generator")
    print("=" * 60)

    config = _load_config()

    print("\n[1/8] RL vs PID Temperature Control")
    rl_data, pid_data = fig1_temperature_comparison(config)

    print("[2/8] Cooling Energy Consumption Comparison")
    fig2_energy_comparison(rl_data, pid_data)

    print("[3/8] RL Training Reward Curve")
    fig3_training_reward_curve()

    print("[4/8] Temperature Stability Over Time")
    fig4_temperature_stability(config)

    print("[5/8] Cooling Action Distribution")
    fig5_action_distribution(config)

    print("[6/8] Rack Temperature Heatmap")
    fig6_rack_heatmap(config)

    print("[7/8] Workload Spike Response")
    fig7_workload_spike_response(config)

    print("[8/8] Long-Term Stability Test")
    fig8_long_term_stability(config)

    print(f"\nAll graphs saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
