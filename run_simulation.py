"""
Run Simulation Script

Runs the data center cooling simulation with different controllers and scenarios.
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator, WorkloadScenario
from workload.dataset_loader import WorkloadTraceLoader
from rl_agent.dqn_agent import DQNAgent
from controllers.pid_controller import PIDController, AdaptivePIDController
from safety.safety_override import SafetyOverride
from evaluation.metrics import CoolingMetrics
from evaluation.experiments import ExperimentRunner


def run_single_simulation(
    controller_type: str,
    config: dict,
    checkpoint_path: str = None,
    num_steps: int = 500,
    workload_pattern: str = "mixed",
    scenario: str = None,
    visualize: bool = True
):
    """
    Run a single simulation episode.
    
    Args:
        controller_type: 'rl', 'pid', or 'adaptive_pid'
        config: Configuration dictionary
        checkpoint_path: Path to RL checkpoint (if using RL)
        num_steps: Number of simulation steps
        workload_pattern: Workload generation pattern
        scenario: Special scenario to test
        visualize: Whether to generate visualizations
    """
    print(f"\n{'='*70}")
    print(f"Running {controller_type.upper()} Controller Simulation")
    print(f"{'='*70}")
    
    # Initialize workload generator
    grid_size = tuple(config['simulation']['grid_size'])
    workload_gen = SyntheticWorkloadGenerator(
        grid_size=grid_size,
        pattern=workload_pattern,
        base_load=config['workload']['base_load'],
        peak_load=config['workload']['peak_load']
    )
    
    # Initialize environment
    env = DataCenterThermalEnv(workload_generator=workload_gen)
    
    # Initialize safety system
    safety = SafetyOverride(
        max_temperature=config['safety']['max_temperature'],
        critical_temperature=config['safety']['critical_temperature']
    )
    
    # Initialize controller
    if controller_type == 'rl':
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        controller = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config['rl']['hidden_dim']
        )
        if checkpoint_path and os.path.exists(checkpoint_path):
            controller.load_checkpoint(checkpoint_path)
            print(f"Loaded RL checkpoint: {checkpoint_path}")
        else:
            print("Warning: No checkpoint loaded. Using untrained agent.")
    
    elif controller_type == 'pid':
        controller = PIDController(
            kp=config['pid']['kp'],
            ki=config['pid']['ki'],
            kd=config['pid']['kd'],
            setpoint=config['pid']['setpoint']
        )
    
    elif controller_type == 'adaptive_pid':
        controller = AdaptivePIDController(
            kp=config['pid']['kp'],
            ki=config['pid']['ki'],
            kd=config['pid']['kd'],
            setpoint=config['pid']['setpoint']
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Reset environment
    state, _ = env.reset()

    # Apply scenario after reset so workload injection is not overwritten.
    if scenario:
        print(f"Applying scenario: {scenario}")
        if scenario == "hotspot":
            env.cpu_workload = WorkloadScenario.create_hotspot_scenario(grid_size)
        elif scenario == "edge_heavy":
            env.cpu_workload = WorkloadScenario.create_edge_heavy_scenario(grid_size)
        elif scenario == "gradient":
            env.cpu_workload = WorkloadScenario.create_gradient_scenario(grid_size)
    
    # History tracking
    temperature_history = []
    cooling_history = []
    workload_history = []
    reward_history = []
    violation_count = 0
    
    print(f"\nRunning simulation for {num_steps} steps...")
    
    # Run simulation
    for step in range(num_steps):
        # Get current state
        state_grids = env.get_state_grid()
        temperatures = state_grids['temperatures']
        cooling_levels = state_grids['cooling_levels']
        workload = state_grids['cpu_workload']
        
        # Store history
        temperature_history.append(temperatures.copy())
        cooling_history.append(cooling_levels.copy())
        workload_history.append(workload.copy())
        
        # Select action
        if controller_type == 'rl':
            action = controller.select_action(state, training=False)
        else:
            # PID controllers
            proposed_cooling = controller.compute(temperatures)
            
            # Apply safety
            safety_status = safety.check_safety(temperatures, cooling_levels, proposed_cooling)
            if safety_status['override_needed']:
                proposed_cooling = safety.apply_override(temperatures, proposed_cooling)
            
            # Set cooling directly
            env.cooling_levels = np.clip(proposed_cooling, 0.0, 1.0)
            action = 1  # PID maintain action
        
        # Step environment
        state, reward, terminated, truncated, info = env.step(action)
        reward_history.append(reward)
        
        # Track violations
        if info['hotspots'] > 0:
            violation_count += info['hotspots']
        
        # Progress indicator
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{num_steps}: "
                  f"Avg Temp = {info['avg_temperature']:.2f}°C, "
                  f"Violations = {violation_count}")
        
        if terminated:
            print(f"Simulation terminated at step {step}")
            break
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = CoolingMetrics.compute_comprehensive_metrics(
        temperature_history,
        cooling_history,
        workload_history,
        target_temp=config['reward']['comfort_zone_temp'],
        max_temp=config['safety']['max_temperature']
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("SIMULATION RESULTS")
    print(f"{'='*70}")
    print(f"Controller: {controller_type.upper()}")
    print(f"\nEnergy Metrics:")
    print(f"  Average Cooling: {metrics['energy']['avg_cooling_level']:.3f}")
    print(f"  Peak Cooling: {metrics['energy']['peak_cooling_level']:.3f}")
    print(f"  Total Energy: {metrics['energy']['total_energy']:.2f} kWh")
    print(f"\nTemperature Metrics:")
    print(f"  Average: {metrics['temperature']['avg_temperature']:.2f}°C")
    print(f"  Max: {metrics['temperature']['max_temperature']:.2f}°C")
    print(f"  Deviation from Target: {metrics['temperature']['temp_deviation']:.2f}°C")
    print(f"  Violations: {metrics['temperature']['violations']}")
    print(f"  Comfort Zone Ratio: {metrics['temperature']['comfort_zone_ratio']*100:.1f}%")
    print(f"\nStability Metrics:")
    print(f"  Avg Temperature Change: {metrics['stability']['avg_temp_change']:.3f}°C")
    print(f"  Settling Time: {metrics['stability']['settling_time']} steps")
    print(f"\nHotspot Metrics:")
    print(f"  Max Hotspots: {metrics['hotspots']['max_hotspots']}")
    print(f"  Hotspot Ratio: {metrics['hotspots']['hotspot_ratio']*100:.1f}%")

    # Store cooling history in metrics for energy comparison
    metrics['_cooling_history'] = cooling_history
    
    # Visualize if requested
    if visualize:
        visualize_results(
            temperature_history,
            cooling_history,
            workload_history,
            reward_history,
            controller_type,
            config
        )
    
    return metrics


def visualize_results(
    temperature_history,
    cooling_history,
    workload_history,
    reward_history,
    controller_type,
    config
):
    """Generate visualization plots."""
    print("\nGenerating visualizations...")
    
    # Create output directory
    output_dir = f"plots/simulation_{controller_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy arrays
    temps = np.array(temperature_history)
    cooling = np.array(cooling_history)
    workload = np.array(workload_history)
    
    # Compute means over time
    temp_mean = np.mean(temps, axis=(1, 2))
    temp_max = np.max(temps, axis=(1, 2))
    cooling_mean = np.mean(cooling, axis=(1, 2))
    workload_mean = np.mean(workload, axis=(1, 2))
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{controller_type.upper()} Controller Simulation Results', fontsize=16)
    
    # Temperature over time
    axes[0, 0].plot(temp_mean, label='Average', linewidth=2)
    axes[0, 0].plot(temp_max, label='Maximum', linewidth=2, alpha=0.7)
    axes[0, 0].axhline(
        y=config['reward']['comfort_zone_temp'],
        color='g', linestyle='--', label='Target'
    )
    axes[0, 0].axhline(
        y=config['safety']['max_temperature'],
        color='r', linestyle='--', label='Max Safe'
    )
    axes[0, 0].set_title('Temperature Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Cooling over time
    axes[0, 1].plot(cooling_mean, linewidth=2)
    axes[0, 1].set_title('Average Cooling Level')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Cooling Level')
    axes[0, 1].grid(True)
    
    # Workload over time
    axes[0, 2].plot(workload_mean, linewidth=2, color='orange')
    axes[0, 2].set_title('Average CPU Workload')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Workload')
    axes[0, 2].grid(True)
    
    # Reward over time
    axes[1, 0].plot(reward_history, linewidth=2, color='purple')
    axes[1, 0].set_title('Reward Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True)
    
    # Final temperature heatmap
    im1 = axes[1, 1].imshow(temps[-1], cmap='hot', aspect='auto')
    axes[1, 1].set_title('Final Temperature Distribution')
    axes[1, 1].set_xlabel('Rack Column')
    axes[1, 1].set_ylabel('Rack Row')
    plt.colorbar(im1, ax=axes[1, 1], label='Temperature (°C)')
    
    # Final cooling heatmap
    im2 = axes[1, 2].imshow(cooling[-1], cmap='Blues', aspect='auto')
    axes[1, 2].set_title('Final Cooling Distribution')
    axes[1, 2].set_xlabel('Rack Column')
    axes[1, 2].set_ylabel('Rack Row')
    plt.colorbar(im2, ax=axes[1, 2], label='Cooling Level')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'simulation_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {output_dir}")


def main():
    """Main simulation function."""
    parser = argparse.ArgumentParser(description="Run data center cooling simulation")
    parser.add_argument(
        '--controller',
        type=str,
        choices=['rl', 'pid', 'adaptive_pid', 'compare'],
        default='rl',
        help='Controller type to use'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/dqn_final.pth',
        help='Path to RL checkpoint (for RL controller)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--workload',
        type=str,
        choices=['sinusoidal', 'spikes', 'burst', 'mixed'],
        default='mixed',
        help='Workload pattern'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['none', 'hotspot', 'edge_heavy', 'gradient'],
        default='none',
        help='Test scenario'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    scenario = None if args.scenario == 'none' else args.scenario
    
    if args.controller == 'compare':
        # Run comparison experiment
        print("\nRunning comparison experiment...")
        runner = ExperimentRunner(config_path=args.config)
        results = runner.run_comparison_experiment(
            rl_checkpoint=args.checkpoint,
            workload_pattern=args.workload,
            num_episodes=5
        )
        print("\nComparison complete!")
        print(f"Results saved to: {results['experiment_dir']}")

        # Also compute energy saved %
        print("\nRunning RL vs PID energy comparison...")
        rl_metrics = run_single_simulation(
            controller_type='rl', config=config,
            checkpoint_path=args.checkpoint, num_steps=args.steps,
            workload_pattern=args.workload, scenario=scenario, visualize=False)
        pid_metrics = run_single_simulation(
            controller_type='pid', config=config,
            num_steps=args.steps, workload_pattern=args.workload,
            scenario=scenario, visualize=False)

        energy_result = CoolingMetrics.compute_energy_saved(
            rl_metrics['_cooling_history'], pid_metrics['_cooling_history'])
        print(f"\n{'='*70}")
        print("ENERGY SAVINGS SUMMARY")
        print(f"{'='*70}")
        print(f"  RL Cooling Energy:       {energy_result['rl_avg_energy']:.2f} units")
        print(f"  PID Cooling Energy:      {energy_result['baseline_avg_energy']:.2f} units")
        print(f"  Energy Saved:            {energy_result['energy_saved_percent']:.2f}%")
    else:
        # Run single simulation
        run_single_simulation(
            controller_type=args.controller,
            config=config,
            checkpoint_path=args.checkpoint,
            num_steps=args.steps,
            workload_pattern=args.workload,
            scenario=scenario,
            visualize=not args.no_viz
        )


if __name__ == "__main__":
    main()
