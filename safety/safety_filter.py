"""
Safety Filter for Data Center Cooling

Intercepts all cooling actions before they reach the simulator.
Guarantees thermal safety regardless of controller output.
The RL controller MUST NOT bypass this layer.
"""

import numpy as np
from typing import Dict, Any, Tuple


# Maximum cooling level constant
MAX_COOLING = 1.0


class SafetyFilter:
    """
    Strict safety filter that sits between controllers and the environment.

    Pipeline:
        Controller (RL/PID) → SafetyFilter → Environment

    Thresholds:
        Target:    60°C   — desired operating temperature
        Warning:   70°C   — gradually increase cooling
        Hotspot:   75°C   — enforce cooling ≥ 0.5
        Critical:  80°C   — force maximum cooling
        Emergency: 85°C   — full shutdown-level cooling
    """

    TARGET_TEMP = 60.0
    WARNING_TEMP = 70.0
    HOTSPOT_TEMP = 75.0
    CRITICAL_TEMP = 80.0
    EMERGENCY_TEMP = 85.0

    def __init__(self):
        self.override_active = False
        self.current_risk = "normal"
        self.override_count = 0

    def apply(
        self,
        cooling_levels: np.ndarray,
        temperature_grid: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply safety filter to proposed cooling levels.

        Args:
            cooling_levels: Proposed cooling from controller [rows, cols]
            temperature_grid: Current temperature grid [rows, cols]

        Returns:
            safe_cooling: Filtered cooling levels [rows, cols]
            filter_info: Metadata about filter actions taken
        """
        safe_cooling = cooling_levels.copy()
        max_temp = float(np.max(temperature_grid))
        was_overridden = False

        # --- Emergency (>85°C): absolute maximum ---
        emergency_mask = temperature_grid > self.EMERGENCY_TEMP
        if np.any(emergency_mask):
            safe_cooling[emergency_mask] = MAX_COOLING
            was_overridden = True

        # --- Critical (>80°C): force maximum cooling on affected racks ---
        critical_mask = (temperature_grid > self.CRITICAL_TEMP) & ~emergency_mask
        if np.any(critical_mask):
            safe_cooling[critical_mask] = MAX_COOLING
            was_overridden = True

        # --- Hotspot (>75°C): enforce cooling ≥ 0.5 ---
        hotspot_mask = (temperature_grid > self.HOTSPOT_TEMP) & ~critical_mask & ~emergency_mask
        if np.any(hotspot_mask):
            safe_cooling[hotspot_mask] = np.maximum(
                safe_cooling[hotspot_mask], 0.5
            )
            was_overridden = True

        # --- Warning (>70°C): gradually increase cooling ---
        warning_mask = (
            (temperature_grid > self.WARNING_TEMP)
            & ~hotspot_mask & ~critical_mask & ~emergency_mask
        )
        if np.any(warning_mask):
            # Proportional increase: 0.35 at 70°C → 0.5 at 75°C
            excess = (temperature_grid[warning_mask] - self.WARNING_TEMP) / 5.0
            min_cooling = np.clip(0.35 + 0.15 * excess, 0.35, 0.50)
            safe_cooling[warning_mask] = np.maximum(
                safe_cooling[warning_mask], min_cooling
            )
            was_overridden = True

        # Determine risk level
        if max_temp > self.CRITICAL_TEMP:
            risk = "emergency"
        elif max_temp > self.HOTSPOT_TEMP:
            risk = "high_risk"
        elif max_temp > self.WARNING_TEMP:
            risk = "moderate_risk"
        else:
            risk = "normal"

        self.current_risk = risk
        self.override_active = was_overridden
        if was_overridden:
            self.override_count += 1

        # Ensure valid range
        safe_cooling = np.clip(safe_cooling, 0.0, MAX_COOLING)

        hotspot_count = int(np.sum(temperature_grid > self.HOTSPOT_TEMP))
        violation_count = int(np.sum(temperature_grid > self.CRITICAL_TEMP))

        filter_info = {
            "risk_level": risk,
            "override_active": was_overridden,
            "override_count": self.override_count,
            "max_temperature": max_temp,
            "hotspot_count": hotspot_count,
            "violation_count": violation_count,
        }

        return safe_cooling, filter_info

    def reset(self):
        """Reset filter state between episodes."""
        self.override_active = False
        self.current_risk = "normal"
        self.override_count = 0
