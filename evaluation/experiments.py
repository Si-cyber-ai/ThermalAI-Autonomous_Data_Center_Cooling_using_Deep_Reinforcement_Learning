"""
Experimental Evaluation Framework

Runs comparative experiments between RL and PID controllers.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import yaml
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator, WorkloadScenario
from rl_agent.dqn_agent import DQNAgent
from controllers.pid_controller import PIDController
from evaluation.metrics import CoolingMetrics, compare_controllers
from safety.safety_override import SafetyOverride


class ExperimentRunner:
    """
    Runs controlled experiments to evaluate cooling controllers.
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        output_dir: str = "experiments"
    ):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory to save experiment results
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        grid_size = tuple(self.config['simulation']['grid_size'])
        self.grid_size = grid_size
        
    def run_controller_episode(
        self,
        controller_type: str,
        workload_generator,
        controller=None,
        num_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run one episode with specified controller.
        
        Args:
            controller_type: 'rl' or 'pid'
            workload_generator: Workload generator instance
            controller: Controller instance (RL agent or PID)
            num_steps: Number of steps (uses config if None)
            
        Returns:
            Episode results dictionary
        """
        if num_steps is None:
            num_steps = self.config['simulation']['max_steps']
        
        # Create environment
        env = DataCenterThermalEnv(
            workload_generator=workload_generator
        )
        
        # Initialize safety system
        safety = SafetyOverride(
            max_temperature=self.config['safety']['max_temperature'],
            critical_temperature=self.config['safety']['critical_temperature'],
            temperature_warning=self.config['safety']['temperature_warning'],
            max_cooling_change=self.config['safety']['max_cooling_change']
        )
        
        # Reset environment
        state, _ = env.reset()
        
        # History tracking
        temperature_history = []
        cooling_history = []
        workload_history = []
        reward_history = []
        
        for step in range(num_steps):
            # Get current state grids
            state_grids = env.get_state_grid()
            temperatures = state_grids['temperatures']
            cooling_levels = state_grids['cooling_levels']
            workload = state_grids['cpu_workload']
            
            # Store history
            temperature_history.append(temperatures.copy())
            cooling_history.append(cooling_levels.copy())
            workload_history.append(workload.copy())
            
            # Select action based on controller type
            if controller_type == 'rl':
                action = controller.select_action(state, training=False)
            elif controller_type == 'pid':
                # PID computes cooling directly
                proposed_cooling = controller.compute(temperatures)
                
                # Apply rate limiting
                proposed_cooling = safety.limit_cooling_rate(
                    cooling_levels, proposed_cooling
                )
                
                # Check for safety override
                safety_status = safety.check_safety(
                    temperatures, cooling_levels, proposed_cooling
                )
                
                if safety_status['override_needed']:
                    proposed_cooling = safety.apply_override(
                        temperatures, proposed_cooling
                    )
                
                # Manually set cooling in environment (bypassing action)
                env.cooling_levels = proposed_cooling
                env.prev_cooling_levels = proposed_cooling.copy()
                
                # Still need to step environment
                action = 1  # PID maintain action
            else:
                raise ValueError(f"Unknown controller type: {controller_type}")
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            reward_history.append(reward)
            
            if terminated:
                print(f"Episode terminated early at step {step}")
                break
        
        # Compute metrics
        metrics = CoolingMetrics.compute_comprehensive_metrics(
            temperature_history,
            cooling_history,
            workload_history,
            target_temp=self.config['reward']['comfort_zone_temp'],
            max_temp=self.config['safety']['max_temperature'],
            timestep=self.config['simulation']['timestep']
        )
        
        results = {
            'controller_type': controller_type,
            'metrics': metrics,
            'history': {
                'temperature': temperature_history,
                'cooling': cooling_history,
                'workload': workload_history,
                'reward': reward_history
            },
            'safety_report': safety.get_status_report()
        }
        
        return results
    
    def run_comparison_experiment(
        self,
        rl_checkpoint: str,
        workload_pattern: str = "mixed",
        num_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Run comparison experiment between RL and PID.
        
        Args:
            rl_checkpoint: Path to trained RL checkpoint
            workload_pattern: Workload pattern type
            num_episodes: Number of evaluation episodes
            
        Returns:
            Comparison results
        """
        print(f"\nRunning comparison experiment...")
        print(f"Workload pattern: {workload_pattern}")
        print(f"Episodes: {num_episodes}")
        
        # Initialize workload generator
        workload_gen = SyntheticWorkloadGenerator(
            grid_size=self.grid_size,
            pattern=workload_pattern,
            base_load=self.config['workload']['base_load'],
            peak_load=self.config['workload']['peak_load']
        )
        
        # Initialize controllers
        # RL Agent
        env_temp = DataCenterThermalEnv(workload_generator=workload_gen)
        state_dim = env_temp.observation_space.shape[0]
        action_dim = env_temp.action_space.n
        
        rl_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config['rl']['hidden_dim']
        )
        
        if os.path.exists(rl_checkpoint):
            rl_agent.load_checkpoint(rl_checkpoint)
        else:
            print(f"Warning: RL checkpoint not found: {rl_checkpoint}")
        
        # PID Controller
        pid_controller = PIDController(
            kp=self.config['pid']['kp'],
            ki=self.config['pid']['ki'],
            kd=self.config['pid']['kd'],
            setpoint=self.config['pid']['setpoint']
        )
        
        # Run episodes
        rl_results_list = []
        pid_results_list = []
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Reset workload generator
            workload_gen.reset()
            
            # Run RL episode
            print("  Running RL controller...")
            rl_results = self.run_controller_episode(
                'rl', workload_gen, controller=rl_agent
            )
            rl_results_list.append(rl_results)
            
            # Reset workload generator
            workload_gen.reset()
            
            # Reset PID
            pid_controller.reset()
            
            # Run PID episode
            print("  Running PID controller...")
            pid_results = self.run_controller_episode(
                'pid', workload_gen, controller=pid_controller
            )
            pid_results_list.append(pid_results)
        
        # Aggregate metrics
        rl_metrics_avg = self._aggregate_metrics(rl_results_list)
        pid_metrics_avg = self._aggregate_metrics(pid_results_list)
        
        # Compare
        comparison_df = compare_controllers(rl_metrics_avg, pid_metrics_avg)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.output_dir, f"experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save comparison table
        comparison_df.to_csv(
            os.path.join(experiment_dir, "comparison.csv"),
            index=False
        )
        
        # Save detailed results
        with open(os.path.join(experiment_dir, "rl_metrics.json"), 'w') as f:
            json.dump(self._jsonify_metrics(rl_metrics_avg), f, indent=2)
        
        with open(os.path.join(experiment_dir, "pid_metrics.json"), 'w') as f:
            json.dump(self._jsonify_metrics(pid_metrics_avg), f, indent=2)
        
        # Generate plots
        self._plot_comparison(
            rl_results_list[0], pid_results_list[0], experiment_dir
        )
        
        print(f"\nResults saved to: {experiment_dir}")
        print("\nComparison Summary:")
        print(comparison_df.to_string(index=False))
        
        return {
            'rl_metrics': rl_metrics_avg,
            'pid_metrics': pid_metrics_avg,
            'comparison': comparison_df,
            'experiment_dir': experiment_dir
        }
    
    def _aggregate_metrics(
        self,
        results_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple episodes.
        
        Args:
            results_list: List of episode results
            
        Returns:
            Averaged metrics
        """
        # Collect all metrics
        all_metrics = [r['metrics'] for r in results_list]
        
        # Average each metric category
        aggregated = {}
        for category in all_metrics[0].keys():
            aggregated[category] = {}
            for metric_name in all_metrics[0][category].keys():
                values = [m[category][metric_name] for m in all_metrics]
                aggregated[category][metric_name] = np.mean(values)
        
        return aggregated
    
    def _jsonify_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types to Python types for JSON serialization."""
        jsonified = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                jsonified[key] = self._jsonify_metrics(value)
            elif isinstance(value, (np.integer, np.floating)):
                jsonified[key] = float(value)
            else:
                jsonified[key] = value
        return jsonified
    
    def _plot_comparison(
        self,
        rl_results: Dict[str, Any],
        pid_results: Dict[str, Any],
        output_dir: str
    ):
        """
        Generate comparison plots.
        
        Args:
            rl_results: RL episode results
            pid_results: PID episode results
            output_dir: Directory to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temperature comparison
        rl_temps = np.mean(rl_results['history']['temperature'], axis=(1, 2))
        pid_temps = np.mean(pid_results['history']['temperature'], axis=(1, 2))
        
        axes[0, 0].plot(rl_temps, label='RL', linewidth=2)
        axes[0, 0].plot(pid_temps, label='PID', linewidth=2)
        axes[0, 0].axhline(
            y=self.config['reward']['comfort_zone_temp'],
            color='g', linestyle='--', label='Target'
        )
        axes[0, 0].set_title('Average Temperature Over Time')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Cooling comparison
        rl_cooling = np.mean(rl_results['history']['cooling'], axis=(1, 2))
        pid_cooling = np.mean(pid_results['history']['cooling'], axis=(1, 2))
        
        axes[0, 1].plot(rl_cooling, label='RL', linewidth=2)
        axes[0, 1].plot(pid_cooling, label='PID', linewidth=2)
        axes[0, 1].set_title('Average Cooling Level Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Cooling Level')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Rewards comparison
        axes[1, 0].plot(rl_results['history']['reward'], label='RL', linewidth=2)
        axes[1, 0].plot(pid_results['history']['reward'], label='PID', linewidth=2)
        axes[1, 0].set_title('Reward Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Metrics comparison (bar chart)
        energy_rl = rl_results['metrics']['energy']['avg_cooling_level']
        energy_pid = pid_results['metrics']['energy']['avg_cooling_level']
        temp_dev_rl = rl_results['metrics']['temperature']['temp_deviation']
        temp_dev_pid = pid_results['metrics']['temperature']['temp_deviation']
        
        categories = ['Energy\nConsumption', 'Temperature\nDeviation']
        rl_values = [energy_rl, temp_dev_rl]
        pid_values = [energy_pid, temp_dev_pid]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, rl_values, width, label='RL')
        axes[1, 1].bar(x + width/2, pid_values, width, label='PID')
        axes[1, 1].set_title('Key Metrics Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend()
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_plots.png'), dpi=150)
        plt.close()
