"""
Digital Twin Thermal Environment for Data Center Simulation

This module implements a Gymnasium-compatible environment for simulating
data center thermal dynamics with reinforcement learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import yaml
import random

from simulator.heat_transfer_model import HeatTransferModel


class DataCenterThermalEnv(gym.Env):
    """
    Data Center Thermal Environment for RL training.
    
    Simulates a 2D grid of server racks with thermal dynamics,
    workload variations, and cooling control.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        workload_generator=None
    ):
        """
        Initialize the data center thermal environment.
        
        Args:
            config_path: Path to configuration file
            workload_generator: Optional workload generator instance
        """
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Grid dimensions
        self.rows, self.cols = self.config['simulation']['grid_size']
        self.num_racks = self.rows * self.cols
        
        # Simulation parameters
        self.timestep = self.config['simulation']['timestep']
        self.max_steps = self.config['simulation']['max_steps']
        self.ambient_temp = self.config['simulation']['ambient_temperature']
        
        # Safety parameters
        self.max_temp = self.config['safety']['max_temperature']
        self.min_temp = self.config['safety']['min_temperature']
        self.critical_temp = self.config['safety']['critical_temperature']
        self.max_cooling_change = self.config['safety']['max_cooling_change']
        
        # Initialize heat transfer model
        self.heat_model = HeatTransferModel(
            grid_size=(self.rows, self.cols),
            alpha=self.config['simulation']['alpha'],
            beta=self.config['simulation']['beta'],
            gamma=self.config['simulation']['gamma'],
            delta=self.config['simulation']['delta'],
            noise_std=self.config['simulation']['noise_std']
        )
        
        # Workload generator
        self.workload_generator = workload_generator
        
        # Define action space (discrete cooling levels)
        # Actions: 0=decrease cooling, 1=maintain, 2=increase low, 3=increase med, 4=increase high
        self.action_space = spaces.Discrete(self.config['rl']['action_dim'])
        
        # Define observation space
        # State includes: temperatures, cpu_workload, cooling_levels, ambient_temp
        state_size = self.num_racks * 3 + 1  # temp + workload + cooling + ambient
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.temperatures: Optional[np.ndarray] = None
        self.cpu_workload: Optional[np.ndarray] = None
        self.cooling_levels: Optional[np.ndarray] = None
        self.current_step: int = 0
        
        # History for tracking
        self.temperature_history = []
        self.cooling_history = []
        self.workload_history = []
        self.reward_history = []
        self.violation_count = 0
        
        # Previous cooling for rate limiting
        self.prev_cooling_levels: Optional[np.ndarray] = None

        # Delayed cooling: stores the effective cooling from the last step.
        # Real CRAC/CRAH units do not respond instantaneously; this state variable
        # carries the previous step's effective cooling so that the blended
        # "effective_cooling" (0.7 * prev + 0.3 * current) can be computed.
        self.prev_effective_cooling: Optional[np.ndarray] = None

        # Step-level cooling diagnostics for evaluation/debugging
        self.last_policy_cooling: float = 0.0
        self.last_final_cooling: float = 0.0
        self.last_safety_added_cooling: float = 0.0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Initialize temperatures near ambient
        self.temperatures = np.random.uniform(
            self.ambient_temp + 10,
            self.ambient_temp + 20,
            size=(self.rows, self.cols)
        )
        
        # Initialize CPU workload
        if self.workload_generator is not None:
            self.cpu_workload = self.workload_generator.generate(0)
        else:
            self.cpu_workload = np.random.uniform(0.2, 0.5, size=(self.rows, self.cols))
        
        # Initialize cooling at moderate level
        self.cooling_levels = np.full((self.rows, self.cols), 0.5)
        self.prev_cooling_levels = self.cooling_levels.copy()
        # Initialise delayed-cooling state to match the starting cooling level
        self.prev_effective_cooling = self.cooling_levels.copy()
        self.last_policy_cooling = float(np.mean(self.cooling_levels))
        self.last_final_cooling = self.last_policy_cooling
        self.last_safety_added_cooling = 0.0
        
        # Reset counters
        self.current_step = 0
        self.violation_count = 0
        
        # Clear history
        self.temperature_history = [self.temperatures.copy()]
        self.cooling_history = [self.cooling_levels.copy()]
        self.workload_history = [self.cpu_workload.copy()]
        self.reward_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Cooling action to take
            
        Returns:
            observation: New state
            reward: Reward for this transition
            terminated: Whether episode has ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Map action to cooling adjustment
        cooling_change = self._action_to_cooling_change(action)
        
        # Apply cooling change with rate limiting
        new_cooling = np.clip(
            self.cooling_levels + cooling_change,
            0.0,
            1.0
        )
        
        # Enforce maximum cooling change rate (safety constraint)
        cooling_delta = new_cooling - self.prev_cooling_levels
        cooling_delta = np.clip(
            cooling_delta,
            -self.max_cooling_change,
            self.max_cooling_change
        )
        self.cooling_levels = self.prev_cooling_levels + cooling_delta

        # Cooling proposed by controller logic before any safety overrides.
        policy_cooling = float(np.mean(self.cooling_levels))
        
        # Cooling floor safety: enforce minimum cooling above threshold
        avg_temp = np.mean(self.temperatures) if self.temperatures is not None else 0
        if avg_temp > 72.0:
            floor = np.clip(0.20 + 0.06 * (avg_temp - 72.0), 0.20, 0.40)
            self.cooling_levels = np.maximum(self.cooling_levels, floor)
        
        # --- Proactive cooling: boost cooling on rising temperature gradient ---
        if len(self.temperature_history) >= 1:
            prev_temps = self.temperature_history[-1]
            temp_gradient = self.temperatures - prev_temps
            rising_fast = temp_gradient > 2.0
            if np.any(rising_fast):
                self.cooling_levels[rising_fast] = np.maximum(
                    self.cooling_levels[rising_fast],
                    np.clip(0.3 + 0.1 * temp_gradient[rising_fast], 0.3, 0.8)
                )
        
        # --- Per-rack escalation control ---
        if self.temperatures is not None:
            # 65-70°C: moderate cooling (strengthened)
            mod_mask = (self.temperatures > 65.0) & (self.temperatures <= 70.0)
            if np.any(mod_mask):
                excess = (self.temperatures[mod_mask] - 65.0) / 5.0
                self.cooling_levels[mod_mask] = np.maximum(
                    self.cooling_levels[mod_mask],
                    np.clip(0.35 + 0.15 * excess, 0.35, 0.50)
                )
            # 70-75°C: high cooling (strengthened)
            high_mask = (self.temperatures > 70.0) & (self.temperatures <= 75.0)
            if np.any(high_mask):
                excess = (self.temperatures[high_mask] - 70.0) / 5.0
                self.cooling_levels[high_mask] = np.maximum(
                    self.cooling_levels[high_mask],
                    np.clip(0.50 + 0.15 * excess, 0.50, 0.65)
                )
            # >75°C: strong cooling (hotspot targeting)
            hot_mask = self.temperatures > 75.0
            if np.any(hot_mask):
                self.cooling_levels[hot_mask] = np.maximum(
                    self.cooling_levels[hot_mask], 0.6
                )
            # >80°C: emergency cooling
            emerg_mask = self.temperatures > 80.0
            if np.any(emerg_mask):
                self.cooling_levels[emerg_mask] = 1.0
        
        self.prev_cooling_levels = self.cooling_levels.copy()
        
        # Update CPU workload
        if self.workload_generator is not None:
            self.cpu_workload = self.workload_generator.generate(self.current_step)
        else:
            # Random walk workload bounded to realistic server utilization
            self.cpu_workload = np.random.uniform(0.4, 0.9, size=(self.rows, self.cols))

        # Introduce occasional workload spikes to simulate bursty real traffic.
        workload_spike = 0.0
        if random.random() < 0.1:
            workload_spike = random.uniform(0.05, 0.15)

        # Thermal inertia model: racks/air volumes do not cool instantly.
        # The 0.85/0.15 split creates a slow, realistic temperature response that
        # allows the RL agent to learn anticipatory (predictive) cooling strategies.
        current_temp = self.temperatures.copy()
        heat_generation = self.heat_model.alpha * self.cpu_workload * 100.0
        if workload_spike > 0.0:
            # Workload spike modelled as additional heat generation (see step 4).
            heat_generation += workload_spike * 100.0

        # Delayed cooling effect: real CRAC/CRAH units have a spin-up lag.
        # effective_cooling blends last-step cooling (70%) with the new commanded
        # cooling (30%), creating a response delay that rewards predictive control.
        effective_cooling = 0.7 * self.prev_effective_cooling + 0.3 * self.cooling_levels
        self.prev_effective_cooling = effective_cooling.copy()

        heat_removal = self.heat_model.beta * effective_cooling * 100.0
        ambient_effect = self.heat_model.delta * (self.ambient_temp - current_temp)
        noise = np.random.normal(0, self.heat_model.noise_std, size=current_temp.shape)

        instantaneous_next = current_temp + (heat_generation - heat_removal + ambient_effect + noise)
        self.temperatures = 0.85 * current_temp + 0.15 * instantaneous_next

        # Rack heat propagation: adjacent racks exchange heat (up/down/left/right).
        neighbor_sum = np.zeros_like(self.temperatures)
        neighbor_count = np.zeros_like(self.temperatures)

        neighbor_sum[1:, :] += self.temperatures[:-1, :]
        neighbor_count[1:, :] += 1
        neighbor_sum[:-1, :] += self.temperatures[1:, :]
        neighbor_count[:-1, :] += 1
        neighbor_sum[:, 1:] += self.temperatures[:, :-1]
        neighbor_count[:, 1:] += 1
        neighbor_sum[:, :-1] += self.temperatures[:, 1:]
        neighbor_count[:, :-1] += 1

        neighbor_avg_temp = np.divide(
            neighbor_sum,
            neighbor_count,
            out=self.temperatures.copy(),
            where=neighbor_count > 0,
        )
        self.temperatures = self.temperatures + 0.05 * (neighbor_avg_temp - self.temperatures)

        # Clamp temperatures to realistic bounds [20, 85]
        self.temperatures = np.clip(self.temperatures, 20.0, 85.0)
        
        # --- Post-update safety override: correct any remaining violations ---
        old_cooling = self.cooling_levels.copy()
        critical_mask = self.temperatures > self.critical_temp       # >85°C
        violation_mask = self.temperatures > self.max_temp            # >80°C
        warning_mask = self.temperatures > self.config['safety']['temperature_warning']  # >75°C
        if np.any(critical_mask):
            self.cooling_levels[critical_mask] = 1.0
        if np.any(violation_mask):
            self.cooling_levels[violation_mask] = np.maximum(
                self.cooling_levels[violation_mask], 1.0
            )
        if np.any(warning_mask):
            self.cooling_levels[warning_mask] = np.maximum(
                self.cooling_levels[warning_mask], 0.6
            )
        # Apply incremental cooling correction to temperatures
        additional_cooling = self.cooling_levels - old_cooling
        if np.any(additional_cooling > 0):
            beta = self.config['simulation']['beta']
            self.temperatures -= additional_cooling * beta * 100.0
        self.prev_cooling_levels = self.cooling_levels.copy()

        # Record final cooling diagnostics after all safety adjustments.
        final_cooling = float(np.mean(self.cooling_levels))
        safety_added_cooling = final_cooling - policy_cooling
        self.last_policy_cooling = policy_cooling
        self.last_final_cooling = final_cooling
        self.last_safety_added_cooling = safety_added_cooling
        
        # Check for safety violations
        violations = np.sum(self.temperatures > self.max_temp)
        critical_violations = np.sum(self.temperatures > self.critical_temp)
        self.violation_count += violations
        
        # Compute reward
        reward = self._compute_reward(violations, critical_violations)
        
        # Update history
        self.temperature_history.append(self.temperatures.copy())
        self.cooling_history.append(self.cooling_levels.copy())
        self.workload_history.append(self.cpu_workload.copy())
        self.reward_history.append(reward)
        
        # Check termination conditions
        self.current_step += 1
        terminated = np.any(self.temperatures > self.critical_temp)  # Only terminate on critical
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _action_to_cooling_change(self, action: int) -> np.ndarray:
        """
        Convert discrete action to cooling level change.
        
        Args:
            action: Discrete action index
            
        Returns:
            Cooling change matrix
        """
        # Action mapping:
        # 0: Decrease cooling globally
        # 1: Maintain current cooling
        # 2: Increase cooling slightly
        # 3: Increase cooling moderately
        # 4: Increase cooling significantly
        
        action_map = {
            0: -0.1,   # Decrease
            1: 0.0,    # Maintain
            2: 0.05,   # Slight increase
            3: 0.1,    # Moderate increase
            4: 0.2     # Significant increase
        }
        
        change = action_map.get(action, 0.0)
        
        # Apply change more aggressively to hotter racks
        temperature_normalized = (self.temperatures - self.ambient_temp) / 50.0
        temperature_weight = np.clip(temperature_normalized, 0.5, 1.5)
        
        cooling_change = change * temperature_weight
        
        return cooling_change
    
    def _compute_reward(self, violations: int, critical_violations: int) -> float:
        """
        Compute reward prioritising energy efficiency while maintaining thermal safety.

        Design rationale
        ----------------
        target_temp = 70°C
            Raised from 68 °C so the agent learns to tolerate slightly warmer
            servers and reduces unnecessary cooling demand.

        Temperature penalty — comfort-band approach
            No penalty is applied while avg_temp stays between 60 °C and 75 °C.
            Outside that band the penalty scales with distance so the agent is
            strongly motivated to stay inside the safe operating range without
            being penalised for every fraction of a degree.

            if avg_temp > 75:  penalty = (avg_temp - 75) * 5   (overheating)
            elif avg_temp < 60: penalty = (60 - avg_temp) * 2  (over-cooled)
            else:               penalty = 0                     (comfortable)

        Energy penalty — 3 × cooling²
            Tripled from the previous 1 × cooling² to strongly discourage
            excessive cooling usage and encourage the agent to reduce fan/CRAC
            power whenever temperatures are safely inside the comfort band.

        Safety deduction
            A flat −100 reward is applied whenever any rack exceeds 80 °C,
            preserving hard thermal safety even under reduced cooling pressure.

        Args:
            violations: Number of temperature violations (racks > max_temp)
            critical_violations: Number of critical violations (racks > critical_temp)

        Returns:
            Reward value (float)
        """
        avg_temp      = float(np.mean(self.temperatures))
        max_temp      = float(np.max(self.temperatures))
        cooling_level = float(np.mean(self.cooling_levels))

        # --- 1. Temperature penalty: zero inside the 60–75 °C comfort band -------
        # Asymmetric: overheating is penalised 2.5× harder than over-cooling because
        # server damage is irreversible whereas wasted cooling just wastes energy.
        if avg_temp > 75.0:
            temp_penalty = (avg_temp - 75.0) * 5.0
        elif avg_temp < 60.0:
            temp_penalty = (60.0 - avg_temp) * 2.0
        else:
            temp_penalty = 0.0

        # --- 2. Energy penalty: 3 × cooling² strongly discourages over-cooling ----
        # Quadratic in cooling level so every marginal increase in fan speed is
        # increasingly expensive, pushing the agent toward the minimum safe level.
        energy_penalty = 3.0 * (cooling_level ** 2)

        # --- 3. Base reward -------------------------------------------------------
        reward = -(temp_penalty + energy_penalty)

        # --- 4. Hard safety deduction for overheating racks ----------------------
        # Applied on top of the temperature penalty so critical events dominate.
        if max_temp > 80.0:
            reward -= 100.0

        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state.
        
        Returns:
            Flattened observation array
        """
        temp_flat = self.temperatures.flatten()
        workload_flat = self.cpu_workload.flatten()
        cooling_flat = self.cooling_levels.flatten()
        ambient = np.array([self.ambient_temp])
        
        observation = np.concatenate([
            temp_flat,
            workload_flat,
            cooling_flat,
            ambient
        ]).astype(np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional environment information.
        
        Returns:
            Info dictionary
        """
        return {
            'step': self.current_step,
            'avg_temperature': np.mean(self.temperatures),
            'max_temperature': np.max(self.temperatures),
            'min_temperature': np.min(self.temperatures),
            'avg_cooling': np.mean(self.cooling_levels),
            'policy_cooling': self.last_policy_cooling,
            'final_cooling': self.last_final_cooling,
            'safety_added_cooling': self.last_safety_added_cooling,
            'avg_workload': np.mean(self.cpu_workload),
            'violations': self.violation_count,
            'hotspots': np.sum(self.temperatures > self.max_temp)
        }
    
    def get_state_grid(self) -> Dict[str, np.ndarray]:
        """
        Get current state as 2D grids for visualization.
        
        Returns:
            Dictionary of state grids
        """
        return {
            'temperatures': self.temperatures.copy(),
            'cpu_workload': self.cpu_workload.copy(),
            'cooling_levels': self.cooling_levels.copy()
        }
    
    def apply_safety_override(self) -> None:
        """
        Apply emergency cooling if temperatures exceed safety threshold.
        """
        danger_mask = self.temperatures > self.config['safety']['temperature_warning']
        if np.any(danger_mask):
            # Force maximum cooling on overheating racks
            self.cooling_levels[danger_mask] = 1.0
    
    def render(self, mode: str = 'human'):
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Temperatures:\n{self.temperatures}")
            print(f"CPU Workload:\n{self.cpu_workload}")
            print(f"Cooling Levels:\n{self.cooling_levels}")
            print(f"Avg Temp: {np.mean(self.temperatures):.2f}°C")
            print(f"Max Temp: {np.max(self.temperatures):.2f}°C")
