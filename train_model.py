"""
Main Training Script for DQN Agent
===================================
AI-Based Data Center Cooling Optimization
using Safe Reinforcement Learning with a Digital Twin Simulator.

Usage:
    python train_model.py                  # Train from scratch (800 episodes)
    python train_model.py --episodes 400   # Override episode count
    python train_model.py --evaluate --resume checkpoints/dqn_final.pth
"""

import argparse
import yaml
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend – safe for headless runs
import matplotlib.pyplot as plt
from datetime import datetime

from rl_agent.training_pipeline import TrainingPipeline
from simulator.thermal_environment import DataCenterThermalEnv
from controllers.pid_controller import PIDController
from workload.synthetic_generator import SyntheticWorkloadGenerator


# ---------------------------------------------------------------------------
# Helper: clean old training artifacts
# ---------------------------------------------------------------------------

def _clean_artifacts(checkpoint_dir: str, log_dir: str, plots_dir: str) -> None:
    """
    Delete stale training artifacts so a fresh run does not mix with previous
    results.  Removes the directories listed below and two well-known CSV /
    JSON files, then recreates the directories so subsequent code can write
    into them without extra mkdir calls.

    Directories removed:
        logs/            – per-step CSV logs and episode summaries
        plots/           – PNG training-progress graphs
        training_graphs/ – legacy graph directory (kept for compatibility)
        checkpoints/     – saved model weights (.pth files)

    Files removed:
        training_rewards.csv
        training_metrics.json
    """
    dirs_to_clean = [checkpoint_dir, log_dir, plots_dir, "training_graphs"]
    files_to_clean = ["training_rewards.csv", "training_metrics.json"]

    print("\n" + "=" * 70)
    print("CLEANING OLD TRAINING ARTIFACTS")
    print("=" * 70)

    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Removed directory : {d}/")

    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed file      : {f}")

    # Recreate necessary directories so downstream code can write to them.
    for d in [checkpoint_dir, log_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)
        print(f"  Recreated         : {d}/")

    print("Artifact cleanup complete.\n")


# ---------------------------------------------------------------------------
# Helper: individual training-curve graphs saved to plots/
# ---------------------------------------------------------------------------

def _save_individual_plots(
    episode_rewards,
    episode_avg_temps,
    episode_cooling_costs,
    episode: int,
    plots_dir: str,
) -> None:
    """
    Save four individual training-progress PNG files after each checkpoint.

    Files written (episode-stamped):
        plots/reward_curve_<episode>.png
        plots/temperature_curve_<episode>.png
        plots/cooling_curve_<episode>.png
        plots/energy_curve_<episode>.png

    The graphs update progressively as more episodes complete, so later
    checkpoints always contain the full history up to that point.
    """
    episodes_x = list(range(1, len(episode_rewards) + 1))

    def _save(y_data, title, ylabel, filename, color, hline=None, hline_label=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes_x, y_data, color=color, linewidth=1.5, alpha=0.85)
        # Smoothed trend (rolling mean with window=min(20, len))
        w = min(20, len(y_data))
        if w > 1:
            rolling = np.convolve(y_data, np.ones(w) / w, mode="valid")
            ax.plot(
                range(w, len(y_data) + 1), rolling,
                color=color, linewidth=2.5, label=f"Smoothed (w={w})"
            )
        if hline is not None:
            ax.axhline(hline, color="red", linestyle="--", linewidth=1.2,
                       label=hline_label or f"y={hline}")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        path = os.path.join(plots_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close(fig)

    _save(
        episode_rewards,
        title=f"Episode Reward vs Episode (up to ep {episode})",
        ylabel="Total Reward",
        filename=f"reward_curve_{episode}.png",
        color="steelblue",
    )
    _save(
        episode_avg_temps,
        title=f"Average Temperature vs Episode (up to ep {episode})",
        ylabel="Temperature (°C)",
        filename=f"temperature_curve_{episode}.png",
        color="tomato",
        hline=68.0,
        hline_label="Target 68 °C",
    )
    _save(
        episode_cooling_costs,
        title=f"Cooling Level vs Episode (up to ep {episode})",
        ylabel="Avg Cooling Level",
        filename=f"cooling_curve_{episode}.png",
        color="seagreen",
    )
    # Energy consumption proxy: cooling_level² (proportional to power draw)
    energy = [c ** 2 for c in episode_cooling_costs]
    _save(
        energy,
        title=f"Energy Consumption vs Episode (up to ep {episode})",
        ylabel="Energy (cooling²)",
        filename=f"energy_curve_{episode}.png",
        color="darkorange",
    )


# ---------------------------------------------------------------------------
# Helper: RL vs PID final evaluation
# ---------------------------------------------------------------------------

def _run_pid_episode(config: dict, num_steps: int = 500) -> dict:
    """
    Run one episode under the PID controller and return key metrics.

    The environment is constructed with the same config and workload generator
    used during RL training so the comparison is fair.
    """
    grid_size = tuple(config["simulation"]["grid_size"])
    workload_gen = SyntheticWorkloadGenerator(
        grid_size=grid_size,
        pattern=config["workload"]["synthetic_pattern"],
        base_load=config["workload"]["base_load"],
        peak_load=config["workload"]["peak_load"],
    )
    env = DataCenterThermalEnv(
        config_path="config.yaml",
        workload_generator=workload_gen,
    )
    pid = PIDController(
        kp=config["pid"]["kp"],
        ki=config["pid"]["ki"],
        kd=config["pid"]["kd"],
        setpoint=config["pid"]["setpoint"],
    )

    state, _ = env.reset()
    pid.reset()

    temps, coolings, violations = [], [], []

    for _ in range(num_steps):
        state_grids = env.get_state_grid()
        temperatures = state_grids["temperatures"]
        proposed = pid.compute(temperatures)
        env.cooling_levels = np.clip(proposed, 0.0, 1.0)
        state, _reward, terminated, truncated, info = env.step(1)  # action 1 = maintain

        temps.append(info["avg_temperature"])
        coolings.append(info["avg_cooling"])
        violations.append(info["hotspots"])

        if terminated or truncated:
            break

    return {
        "avg_temp": float(np.mean(temps)),
        "max_temp": float(np.max([info["max_temperature"] if "max_temperature" in info else t
                                   for t in temps])),
        "avg_cooling": float(np.mean(coolings)),
        "avg_energy": float(np.mean([c ** 2 for c in coolings])),
        "violations": int(np.sum(violations)),
    }


def _run_rl_episode(pipeline: "TrainingPipeline", config: dict, num_steps: int = 500) -> dict:
    """
    Run one greedy evaluation episode under the trained RL agent and return metrics.
    """
    env = pipeline.env
    state, _ = env.reset()

    temps, coolings, max_temps, violations = [], [], [], []

    for _ in range(num_steps):
        action = pipeline.agent.select_action(state, training=False)
        state, _reward, terminated, truncated, info = env.step(action)

        temps.append(info["avg_temperature"])
        max_temps.append(info["max_temperature"])
        coolings.append(info["avg_cooling"])
        violations.append(info["hotspots"])

        if terminated or truncated:
            break

    return {
        "avg_temp": float(np.mean(temps)),
        "max_temp": float(np.max(max_temps)) if max_temps else 0.0,
        "avg_cooling": float(np.mean(coolings)),
        "avg_energy": float(np.mean([c ** 2 for c in coolings])),
        "violations": int(np.sum(violations)),
    }


def _final_evaluation(pipeline: "TrainingPipeline", config: dict, num_eval_eps: int = 5):
    """
    Evaluate the trained RL agent and a PID baseline under identical conditions.

    Prints:
        Average temperature, Max temperature, Average cooling level,
        RL avg energy/step, PID avg energy/step, and Energy saving %.

    Energy saving formula:
        EnergySaved = ((PID_energy - RL_energy) / PID_energy) * 100
    Clamped to [-50, 100] to avoid numerical instability.
    """
    print("\n" + "=" * 70)
    print("FINAL EVALUATION: RL vs PID")
    print("=" * 70)

    rl_metrics_list = []
    pid_metrics_list = []

    print(f"Running {num_eval_eps} evaluation episodes for each controller...")
    for ep in range(num_eval_eps):
        print(f"  Episode {ep + 1}/{num_eval_eps}", end="\r")
        rl_m = _run_rl_episode(pipeline, config)
        pid_m = _run_pid_episode(config)
        rl_metrics_list.append(rl_m)
        pid_metrics_list.append(pid_m)

    def _mean(lst, key):
        return float(np.mean([d[key] for d in lst]))

    rl_avg_temp   = _mean(rl_metrics_list, "avg_temp")
    rl_max_temp   = max(d["max_temp"] for d in rl_metrics_list)
    rl_avg_cool   = _mean(rl_metrics_list, "avg_cooling")
    rl_avg_energy = _mean(rl_metrics_list, "avg_energy")
    rl_violations = sum(d["violations"] for d in rl_metrics_list)

    pid_avg_temp   = _mean(pid_metrics_list, "avg_temp")
    pid_max_temp   = max(d["max_temp"] for d in pid_metrics_list)
    pid_avg_cool   = _mean(pid_metrics_list, "avg_cooling")
    pid_avg_energy = _mean(pid_metrics_list, "avg_energy")
    pid_violations = sum(d["violations"] for d in pid_metrics_list)

    # Energy saved percentage (clamped)
    if pid_avg_energy > 1e-9:
        raw_saved = ((pid_avg_energy - rl_avg_energy) / pid_avg_energy) * 100.0
    else:
        raw_saved = 0.0
    energy_saved = max(min(raw_saved, 100.0), -50.0)

    print("\n" + "-" * 70)
    print(f"{'Metric':<35} {'RL Agent':>12} {'PID':>12}")
    print("-" * 70)
    print(f"{'Average Temperature (°C)':<35} {rl_avg_temp:>12.2f} {pid_avg_temp:>12.2f}")
    print(f"{'Max Temperature (°C)':<35} {rl_max_temp:>12.2f} {pid_max_temp:>12.2f}")
    print(f"{'Average Cooling Level':<35} {rl_avg_cool:>12.3f} {pid_avg_cool:>12.3f}")
    print(f"{'Avg Energy/Step (cooling²)':<35} {rl_avg_energy:>12.4f} {pid_avg_energy:>12.4f}")
    print(f"{'Total Violations':<35} {rl_violations:>12d} {pid_violations:>12d}")
    print("-" * 70)
    print(f"\n  RL Avg Energy/Step  : {rl_avg_energy:.4f}")
    print(f"  PID Avg Energy/Step : {pid_avg_energy:.4f}")
    print(f"  Energy Saving       : {energy_saved:.2f}%")

    if energy_saved >= 20:
        print(f"\n✅ RL achieves ~{energy_saved:.1f}% energy savings over PID baseline.")
    elif energy_saved > 0:
        print(f"\n⚠️  RL saves {energy_saved:.1f}% energy – within acceptable range.")
    else:
        print(f"\n⚠️  PID used less energy in this run ({-energy_saved:.1f}% more for RL).")
        print("   Consider re-tuning the reward function or training longer.")

    return {
        "rl": {
            "avg_temp": rl_avg_temp, "max_temp": rl_max_temp,
            "avg_cooling": rl_avg_cool, "avg_energy": rl_avg_energy,
            "violations": rl_violations,
        },
        "pid": {
            "avg_temp": pid_avg_temp, "max_temp": pid_max_temp,
            "avg_cooling": pid_avg_cool, "avg_energy": pid_avg_energy,
            "violations": pid_violations,
        },
        "energy_saved_pct": energy_saved,
    }


# ---------------------------------------------------------------------------
# Patched TrainingPipeline wrapper that saves per-checkpoint individual plots
# ---------------------------------------------------------------------------

class EnhancedTrainingPipeline(TrainingPipeline):
    """
    Extends TrainingPipeline to save individual per-metric PNG files into
    the plots/ directory after each checkpoint (every save_frequency episodes).

    File naming:
        plots/reward_curve_<episode>.png
        plots/temperature_curve_<episode>.png
        plots/cooling_curve_<episode>.png
        plots/energy_curve_<episode>.png
    """

    def __init__(self, config_path: str, checkpoint_dir: str, log_dir: str, plots_dir: str):
        super().__init__(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)

    def train(self, num_episodes=None, save_frequency=None):
        """Override to inject per-checkpoint individual-plot saving."""
        import tqdm as _tqdm

        if num_episodes is None:
            num_episodes = self.config["rl"]["training_episodes"]
        if save_frequency is None:
            save_frequency = self.config["checkpoints"]["save_frequency"]

        print(f"\nStarting DQN training for {num_episodes} episodes...")
        print(f"State dim: {self.agent.state_dim}, Action dim: {self.agent.action_dim}")

        for episode in _tqdm.tqdm(range(num_episodes), desc="Training"):
            episode_reward, episode_info = self._run_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_info["length"])
            self.episode_avg_temps.append(episode_info["avg_temp"])
            self.episode_violations.append(episode_info["violations"])
            self.episode_cooling_costs.append(episode_info["cooling_cost"])

            # Periodically update target network
            if (episode + 1) % self.agent.target_update_freq == 0:
                self.agent.update_target_network()

            # Checkpoint + graphs every save_frequency episodes
            if (episode + 1) % save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"dqn_episode_{episode + 1}.pth"
                )
                self.agent.save_checkpoint(checkpoint_path)

                # Combined progress plot (existing behaviour)
                if self.enable_training_plots:
                    self._save_training_plots(episode + 1)

                # Individual per-metric plots in plots/
                _save_individual_plots(
                    self.episode_rewards,
                    self.episode_avg_temps,
                    self.episode_cooling_costs,
                    episode=episode + 1,
                    plots_dir=self.plots_dir,
                )

            if (episode + 1) % 10 == 0:
                self._log_progress(episode + 1)

        # Final checkpoint + plots
        final_path = os.path.join(self.checkpoint_dir, "dqn_final.pth")
        self.agent.save_checkpoint(final_path)
        if self.enable_training_plots:
            self._save_training_plots(num_episodes)
        _save_individual_plots(
            self.episode_rewards,
            self.episode_avg_temps,
            self.episode_cooling_costs,
            episode=num_episodes,
            plots_dir=self.plots_dir,
        )

        print("\nTraining completed!")
        return self._get_training_summary()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Main training / evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Train DQN agent for data center cooling optimization"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (overrides config; default: 800)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--plots-dir", type=str, default="plots",
                        help="Directory to save training graphs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (skips cleanup)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate trained agent instead of training")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip artifact cleanup before training")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("AI-Based Data Center Cooling Optimization")
    print("Safe Reinforcement Learning with Digital Twin Simulator")
    print("=" * 70)
    print(f"\nConfiguration : {args.config}")
    print(f"Grid Size     : {config['simulation']['grid_size']}")
    print(f"Action Space  : {config['rl']['action_dim']} discrete actions")
    print(f"Workload      : {config['workload']['synthetic_pattern']}")

    # ------------------------------------------------------------------
    # Step 1: Clean old artifacts before fresh training
    # ------------------------------------------------------------------
    if not args.evaluate and not args.resume and not args.no_clean:
        _clean_artifacts(
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            plots_dir=args.plots_dir,
        )

    # ------------------------------------------------------------------
    # Initialise enhanced training pipeline
    # ------------------------------------------------------------------
    pipeline = EnhancedTrainingPipeline(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        plots_dir=args.plots_dir,
    )

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        pipeline.agent.load_checkpoint(args.resume)

    # ------------------------------------------------------------------
    # Evaluation-only mode
    # ------------------------------------------------------------------
    if args.evaluate:
        print("\n" + "=" * 70)
        print("EVALUATION MODE")
        print("=" * 70)

        if not args.resume:
            print("Warning: No checkpoint specified. Using untrained agent.")

        eval_results = pipeline.evaluate(num_episodes=args.eval_episodes, render=False)

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Average Reward     : {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"Average Temperature: {eval_results['avg_temperature']:.2f}°C")
        print(f"Total Violations   : {eval_results['total_violations']}")
        print(f"Success Rate       : {eval_results['success_rate'] * 100:.1f}%")

        # Also run RL vs PID comparison in eval mode
        _final_evaluation(pipeline, config, num_eval_eps=args.eval_episodes)
        return

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING MODE")
    print("=" * 70)

    num_episodes = args.episodes if args.episodes else config["rl"]["training_episodes"]
    save_freq    = config["checkpoints"]["save_frequency"]

    print(f"Episodes       : {num_episodes}")
    print(f"Save frequency : every {save_freq} episodes")
    print(f"Epsilon min    : {config['rl']['epsilon_end']}")
    print(f"Checkpoints    : {args.checkpoint_dir}/")
    print(f"Plots          : {args.plots_dir}/")

    start_time = datetime.now()

    # ------------------------------------------------------------------
    # Train agent (800 episodes by default)
    # ------------------------------------------------------------------
    training_summary = pipeline.train(
        num_episodes=num_episodes,
        save_frequency=save_freq,
    )

    duration = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total Episodes         : {training_summary['total_episodes']}")
    print(f"Training Duration      : {duration / 60:.1f} minutes")
    print(f"Final Avg Reward       : {training_summary['final_avg_reward']:.2f}")
    print(f"Final Avg Temperature  : {training_summary['final_avg_temp']:.2f}°C")
    print(f"Total Violations       : {training_summary['total_violations']}")
    print(f"Best Episode           : {training_summary['best_episode']} "
          f"(Reward: {training_summary['best_reward']:.2f})")
    print(f"Final Epsilon          : {training_summary['final_epsilon']:.4f}")
    print(f"\nFinal model saved to   : {args.checkpoint_dir}/dqn_final.pth")
    print(f"Training plots saved to: {args.plots_dir}/")

    # ------------------------------------------------------------------
    # Step 9: Final evaluation – RL vs PID
    # ------------------------------------------------------------------
    _final_evaluation(pipeline, config, num_eval_eps=5)

    print("\n" + "=" * 70)
    print("All done! Training pipeline completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
