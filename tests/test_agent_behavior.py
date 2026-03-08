"""
Section 3 — Agent Behavior Validation

Forces different temperature states and verifies the CoolingAgent
chooses the correct strategy for each risk level.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from agents.cooling_agent import CoolingAgent


def _make_state(temp_value: float, rows: int = 3, cols: int = 4):
    """Create a synthetic state dict with uniform temperatures."""
    return {
        "temperatures": np.full((rows, cols), temp_value),
        "cooling_levels": np.full((rows, cols), 0.3),
        "cpu_workload": np.full((rows, cols), 0.5),
    }


def run_tests():
    agent = CoolingAgent()

    cases = [
        # (label, temperature, expected_strategy)
        ("Normal Operation", 62.0, "rl_control"),
        ("Moderate Risk", 72.0, "pid_control"),
        ("High Risk", 77.0, "strong_cooling"),
        ("Emergency", 82.0, "max_cooling"),
    ]

    results = []
    print("=" * 60)
    print("Agent Behavior Validation")
    print("=" * 60)

    for label, temp, expected in cases:
        agent.reset()
        state = _make_state(temp)

        risk = agent.evaluate_risk(state)
        strategy = agent.choose_strategy(risk)

        passed = strategy == expected
        status = "PASS" if passed else "FAIL"
        results.append((label, passed))

        print(
            f"  [{status}] {label:20s}  "
            f"temp={temp:.0f}°C  risk={risk:15s}  "
            f"strategy={strategy:15s}  (expected={expected})"
        )

    print("-" * 60)
    all_pass = all(p for _, p in results)
    print(f"{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print()
    return all_pass


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
