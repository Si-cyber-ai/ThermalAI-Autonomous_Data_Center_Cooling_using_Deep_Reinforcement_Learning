"""
Supervisory AI Cooling Agent for Data Center Thermal Management

Sits above the RL and PID controllers as a decision-making supervisor.
Evaluates system risk, selects the best control strategy, and ensures
the safety filter is always applied before actions reach the environment.

Architecture:
    Digital Twin Environment
           ↓
    CoolingAgent (Supervisor)
           ↓
    Controller Layer (RL / PID)
           ↓
    SafetyFilter
           ↓
    Cooling Action → Thermal Simulation
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from safety.safety_filter import SafetyFilter, MAX_COOLING


class CoolingAgent:
    """
    Supervisory AI Cooling Agent.

    Observes the data center state, evaluates thermal risk, selects the
    appropriate control strategy (RL vs PID vs direct), and enforces the
    safety filter on every action.
    """

    # Escalation thresholds
    NORMAL_CEILING = 65.0
    MODERATE_CEILING = 70.0
    HIGH_CEILING = 75.0
    EMERGENCY_CEILING = 80.0

    def __init__(self):
        self.safety_filter = SafetyFilter()
        self._strategy = "rl_control"
        self._risk_level = "normal"
        self._step_count = 0

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def observe(self, state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Pass-through observation (can be extended with preprocessing)."""
        return state

    def evaluate_risk(self, state: Dict[str, np.ndarray]) -> str:
        """
        Evaluate current thermal risk from the temperature grid.

        Returns one of: "normal", "moderate_risk", "high_risk", "emergency"
        """
        temperatures = state["temperatures"]
        max_temp = float(np.max(temperatures))

        if max_temp > self.EMERGENCY_CEILING:
            return "emergency"
        if max_temp > self.HIGH_CEILING:
            return "high_risk"
        if max_temp > self.MODERATE_CEILING:
            return "moderate_risk"
        return "normal"

    def choose_strategy(self, risk: str) -> str:
        """
        Select control strategy based on risk level.

        Returns one of:
            "rl_control"      — allow RL agent to act freely
            "pid_control"     — use PID for stability
            "strong_cooling"  — fixed high cooling
            "max_cooling"     — emergency full cooling
        """
        if risk == "emergency":
            return "max_cooling"
        if risk == "high_risk":
            return "strong_cooling"
        if risk == "moderate_risk":
            return "pid_control"
        return "rl_control"

    def act(
        self,
        env,
        rl_controller,
        pid_controller,
        obs: np.ndarray,
        *,
        training: bool = False,
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Full supervisor step: observe → evaluate risk → choose strategy →
        compute cooling → apply safety filter.

        Args:
            env: DataCenterThermalEnv instance
            rl_controller: DQNAgent
            pid_controller: PIDController
            obs: Current observation vector (for RL)
            training: Whether RL is in training mode

        Returns:
            safe_cooling: Safety-filtered cooling grid [rows, cols]
            rl_action: The discrete action chosen (for env.step)
            agent_info: Metadata dict with risk, strategy, filter info
        """
        state = self.observe(env.get_state_grid())
        temperatures = state["temperatures"]

        risk = self.evaluate_risk(state)
        strategy = self.choose_strategy(risk)

        self._risk_level = risk
        self._strategy = strategy
        self._step_count += 1

        # --- Agent decision logging ---
        print(
            f"[Agent] step={self._step_count} mode={risk} "
            f"strategy={strategy} "
            f"max_temp={float(np.max(temperatures)):.2f}"
        )

        # ----- Compute raw cooling based on strategy -----
        if strategy == "max_cooling":
            raw_cooling = np.full_like(temperatures, MAX_COOLING)
            rl_action = 4  # highest discrete action

        elif strategy == "strong_cooling":
            raw_cooling = np.full_like(temperatures, 0.7)
            # Hotspot targeting: push even harder on hot racks
            hotspot_mask = temperatures > 75.0
            raw_cooling[hotspot_mask] = np.maximum(
                raw_cooling[hotspot_mask], 0.85
            )
            rl_action = 4

        elif strategy == "pid_control":
            raw_cooling = pid_controller.compute(temperatures)
            raw_cooling = np.clip(raw_cooling, 0.0, 1.0)
            rl_action = 1  # "maintain" action code

        else:  # rl_control
            rl_action = rl_controller.select_action(obs, training=training)
            # Let the environment's action mapping handle the RL change;
            # but we still apply the safety filter to the *current* levels
            # after the env processes the action.  Return early — the env
            # will modify cooling_levels via step().  We only need post-step
            # safety filtering.
            raw_cooling = None  # signal: let env handle it

        # ----- Apply safety filter (ALWAYS) -----
        if raw_cooling is not None:
            safe_cooling, filter_info = self.safety_filter.apply(
                raw_cooling, temperatures
            )
            env.cooling_levels = safe_cooling
        else:
            # RL mode: safety filter applied after env.step (post-step)
            filter_info = {
                "risk_level": risk,
                "override_active": False,
                "override_count": self.safety_filter.override_count,
                "max_temperature": float(np.max(temperatures)),
                "hotspot_count": int(np.sum(temperatures > 75.0)),
                "violation_count": int(np.sum(temperatures > 80.0)),
            }

        agent_info = {
            "risk_level": risk,
            "strategy": strategy,
            "step": self._step_count,
            **filter_info,
        }

        return (
            env.cooling_levels.copy() if raw_cooling is not None else None,
            rl_action,
            agent_info,
        )

    def post_step_safety(
        self, env
    ) -> Dict[str, Any]:
        """
        Apply safety filter AFTER env.step() completes.
        Called every step to guarantee no violations persist.

        Returns:
            filter_info dict
        """
        state = env.get_state_grid()
        temperatures = state["temperatures"]
        cooling = state["cooling_levels"]

        safe_cooling, filter_info = self.safety_filter.apply(
            cooling, temperatures
        )

        # Apply incremental correction directly to temperatures
        additional = safe_cooling - env.cooling_levels
        if np.any(additional > 0):
            beta = env.config["simulation"]["beta"]
            env.temperatures -= additional * beta * 100.0

        env.cooling_levels = safe_cooling
        env.prev_cooling_levels = safe_cooling.copy()

        return filter_info

    def reset(self):
        """Reset agent state for a new episode."""
        self.safety_filter.reset()
        self._strategy = "rl_control"
        self._risk_level = "normal"
        self._step_count = 0

    # ------------------------------------------------------------------
    # Read-only properties for dashboard display
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def risk_level(self) -> str:
        return self._risk_level
