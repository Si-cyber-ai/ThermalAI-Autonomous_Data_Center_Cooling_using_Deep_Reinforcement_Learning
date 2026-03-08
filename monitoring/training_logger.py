"""
Training Logger for RL Agent

Stores per-step metrics, writes CSV logs, and provides data for the dashboard.
"""

import csv
import os
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime


class TrainingLogger:
    """
    Logs training metrics per step and per episode for visualization.
    """

    STEP_FIELDS = [
        "episode", "step", "reward", "avg_temperature", "max_temperature",
        "cooling_level", "energy_consumption", "violations", "epsilon", "loss",
        "action"
    ]

    EPISODE_FIELDS = [
        "episode", "total_reward", "avg_temperature", "max_temperature",
        "total_violations", "avg_cooling", "episode_length", "epsilon",
        "avg_loss", "action_distribution"
    ]

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.step_log_path = os.path.join(log_dir, "training_logs.csv")
        self.episode_log_path = os.path.join(log_dir, "episode_logs.csv")

        # In-memory buffers for dashboard access
        self.step_records: List[Dict[str, Any]] = []
        self.episode_records: List[Dict[str, Any]] = []
        self.action_counts: Dict[int, int] = defaultdict(int)

        # Current episode accumulator
        self._episode_steps: List[Dict[str, Any]] = []
        self._episode_actions: List[int] = []

        # Initialize CSV files with headers
        self._init_csv(self.step_log_path, self.STEP_FIELDS)
        self._init_csv(self.episode_log_path, self.EPISODE_FIELDS)

    # ------------------------------------------------------------------
    def _init_csv(self, path: str, fields: List[str]):
        """Create (or overwrite) the CSV with a fresh header."""
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    # ------------------------------------------------------------------
    def log_step(
        self,
        episode: int,
        step: int,
        reward: float,
        avg_temperature: float,
        max_temperature: float,
        cooling_level: float,
        energy_consumption: float,
        violations: int,
        epsilon: float,
        loss: Optional[float],
        action: Optional[int] = None,
    ):
        """Log a single training step."""
        record = {
            "episode": episode,
            "step": step,
            "reward": round(reward, 4),
            "avg_temperature": round(avg_temperature, 4),
            "max_temperature": round(max_temperature, 4),
            "cooling_level": round(cooling_level, 4),
            "energy_consumption": round(energy_consumption, 4),
            "violations": violations,
            "epsilon": round(epsilon, 6),
            "loss": round(loss, 6) if loss is not None else 0.0,
            "action": action if action is not None else -1,
        }

        # Write to CSV
        with open(self.step_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.STEP_FIELDS)
            writer.writerow(record)

        self.step_records.append(record)
        self._episode_steps.append(record)

        if action is not None:
            self.action_counts[action] += 1
            self._episode_actions.append(action)

    # ------------------------------------------------------------------
    def end_episode(self, episode: int, epsilon: float):
        """Summarise and log the completed episode."""
        steps = self._episode_steps
        if not steps:
            return

        # Action distribution as string "0:n,1:n,..."
        act_dist = defaultdict(int)
        for a in self._episode_actions:
            act_dist[a] += 1
        act_str = ",".join(f"{k}:{v}" for k, v in sorted(act_dist.items()))

        losses = [s["loss"] for s in steps if s["loss"] != 0.0]

        record = {
            "episode": episode,
            "total_reward": round(sum(s["reward"] for s in steps), 4),
            "avg_temperature": round(np.mean([s["avg_temperature"] for s in steps]), 4),
            "max_temperature": round(max(s["max_temperature"] for s in steps), 4),
            "total_violations": sum(s["violations"] for s in steps),
            "avg_cooling": round(np.mean([s["cooling_level"] for s in steps]), 4),
            "episode_length": len(steps),
            "epsilon": round(epsilon, 6),
            "avg_loss": round(np.mean(losses), 6) if losses else 0.0,
            "action_distribution": act_str,
        }

        with open(self.episode_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.EPISODE_FIELDS)
            writer.writerow(record)

        self.episode_records.append(record)

        # Reset accumulators
        self._episode_steps = []
        self._episode_actions = []

    # ------------------------------------------------------------------
    # Dashboard helpers
    # ------------------------------------------------------------------
    def get_episode_dataframe(self):
        """Return episode-level metrics as list of dicts (or load from CSV)."""
        if self.episode_records:
            return self.episode_records
        # Fallback: read from CSV
        import pandas as pd
        if os.path.exists(self.episode_log_path):
            return pd.read_csv(self.episode_log_path).to_dict("records")
        return []

    def get_action_distribution(self) -> Dict[int, int]:
        """Return cumulative action counts."""
        return dict(self.action_counts)

    def get_latest_episode(self) -> Optional[Dict[str, Any]]:
        """Return most recent episode summary."""
        return self.episode_records[-1] if self.episode_records else None

    def reset_logs(self):
        """Clear all logs and start fresh."""
        self.step_records.clear()
        self.episode_records.clear()
        self.action_counts.clear()
        self._episode_steps.clear()
        self._episode_actions.clear()
        self._init_csv(self.step_log_path, self.STEP_FIELDS)
        self._init_csv(self.episode_log_path, self.EPISODE_FIELDS)
