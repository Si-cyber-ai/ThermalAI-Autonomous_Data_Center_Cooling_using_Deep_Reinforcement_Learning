"""
Training Pipeline for DQN Agent

Implements training loop with logging, checkpointing, and monitoring.
"""

import numpy as np
import torch
from tqdm import tqdm
import yaml
import os
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime

from rl_agent.dqn_agent import DQNAgent
from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator
from monitoring.training_logger import TrainingLogger


class TrainingPipeline:
    """
    Training pipeline for DQN agent on data center cooling task.
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to configuration file
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        # plots/ directory for individual per-metric training graphs
        self.plots_dir = self.config.get("visualization", {}).get("plot_dir", "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.enable_training_plots = bool(self.config.get("visualization", {}).get("save_plots", True))
        
        # Initialize workload generator
        grid_size = tuple(self.config['simulation']['grid_size'])
        self.workload_generator = SyntheticWorkloadGenerator(
            grid_size=grid_size,
            pattern=self.config['workload']['synthetic_pattern'],
            base_load=self.config['workload']['base_load'],
            peak_load=self.config['workload']['peak_load']
        )
        
        # Initialize environment
        self.env = DataCenterThermalEnv(
            config_path=config_path,
            workload_generator=self.workload_generator
        )
        
        # Initialize DQN agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config['rl']['hidden_dim'],
            learning_rate=self.config['rl']['learning_rate'],
            gamma=self.config['rl']['gamma'],
            epsilon_start=self.config['rl']['epsilon_start'],
            epsilon_end=self.config['rl']['epsilon_end'],
            epsilon_decay=self.config['rl']['epsilon_decay'],
            batch_size=self.config['rl']['batch_size'],
            memory_size=self.config['rl']['memory_size'],
            target_update_freq=self.config['rl']['target_update_freq'],
            entropy_weight=self.config['rl'].get('entropy_weight', 0.01),
            device=device
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_avg_temps = []
        self.episode_violations = []
        self.episode_cooling_costs = []
        
        # Training logger for CSV output and dashboard
        self.logger = TrainingLogger(log_dir=log_dir)
        
    def train(
        self,
        num_episodes: Optional[int] = None,
        save_frequency: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the DQN agent.
        
        Args:
            num_episodes: Number of episodes to train (uses config if None)
            save_frequency: Checkpoint save frequency (uses config if None)
            
        Returns:
            Dictionary of training metrics
        """
        if num_episodes is None:
            num_episodes = self.config['rl']['training_episodes']
        if save_frequency is None:
            save_frequency = self.config['checkpoints']['save_frequency']
        
        print(f"\nStarting DQN training for {num_episodes} episodes...")
        print(f"State dim: {self.agent.state_dim}, Action dim: {self.agent.action_dim}")
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            episode_reward, episode_info = self._run_episode()
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_info['length'])
            self.episode_avg_temps.append(episode_info['avg_temp'])
            self.episode_violations.append(episode_info['violations'])
            self.episode_cooling_costs.append(episode_info['cooling_cost'])
            
            # Update target network periodically
            if (episode + 1) % self.agent.target_update_freq == 0:
                self.agent.update_target_network()
            
            # Save checkpoint periodically
            if (episode + 1) % save_frequency == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"dqn_episode_{episode+1}.pth"
                )
                self.agent.save_checkpoint(checkpoint_path)
                if self.enable_training_plots:
                    self._save_training_plots(episode + 1)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                self._log_progress(episode + 1)
        
        # Final checkpoint
        final_checkpoint = os.path.join(self.checkpoint_dir, "dqn_final.pth")
        self.agent.save_checkpoint(final_checkpoint)
        if self.enable_training_plots:
            self._save_training_plots(num_episodes)
        
        print("\nTraining completed!")
        
        return self._get_training_summary()
    
    def _run_episode(self) -> tuple:
        """
        Run one training episode with per-step logging.
        
        Returns:
            Tuple of (total_reward, episode_info)
        """
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        total_temp = 0.0
        total_violations = 0
        total_cooling = 0.0
        episode_num = self.agent.episodes_done
        
        done = False
        
        while not done:
            # Select action
            action = self.agent.select_action(state, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = self.agent.train_step()
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            total_temp += info['avg_temperature']
            total_violations += info['hotspots']
            total_cooling += info['avg_cooling']
            
            # Log step
            self.logger.log_step(
                episode=episode_num,
                step=episode_length,
                reward=reward,
                avg_temperature=info['avg_temperature'],
                max_temperature=info['max_temperature'],
                cooling_level=info['avg_cooling'],
                energy_consumption=info['avg_cooling'],
                violations=info['hotspots'],
                epsilon=self.agent.epsilon,
                loss=loss,
                action=action,
            )
            
            state = next_state
        
        # End episode logging
        self.logger.end_episode(episode_num, self.agent.epsilon)
        
        episode_info = {
            'length': episode_length,
            'avg_temp': total_temp / episode_length if episode_length > 0 else 0,
            'violations': total_violations,
            'cooling_cost': total_cooling / episode_length if episode_length > 0 else 0
        }
        
        self.agent.episodes_done += 1
        
        return episode_reward, episode_info
    
    def _log_progress(self, episode: int):
        """
        Log training progress.
        
        Args:
            episode: Current episode number
        """
        recent_rewards = self.episode_rewards[-10:]
        recent_temps = self.episode_avg_temps[-10:]
        recent_violations = self.episode_violations[-10:]
        
        stats = self.agent.get_statistics()
        
        print(f"\nEpisode {episode}")
        print(f"  Avg Reward (last 10): {np.mean(recent_rewards):.2f}")
        print(f"  Avg Temperature (last 10): {np.mean(recent_temps):.2f}°C")
        print(f"  Total Violations (last 10): {np.sum(recent_violations)}")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Avg Loss: {stats['avg_loss']:.4f}")
    
    def _save_training_plots(self, episode: int):
        """
        Save training progress plots.
        
        Args:
            episode: Current episode number
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Average temperature
        axes[0, 1].plot(self.episode_avg_temps)
        axes[0, 1].axhline(
            y=self.config['reward']['comfort_zone_temp'],
            color='g',
            linestyle='--',
            label='Target Temp'
        )
        axes[0, 1].axhline(
            y=self.config['safety']['max_temperature'],
            color='r',
            linestyle='--',
            label='Max Temp'
        )
        axes[0, 1].set_title('Average Temperature')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Violations
        axes[1, 0].plot(self.episode_violations)
        axes[1, 0].set_title('Temperature Violations per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Number of Violations')
        axes[1, 0].grid(True)
        
        # Cooling cost
        axes[1, 1].plot(self.episode_cooling_costs)
        axes[1, 1].set_title('Average Cooling Level')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cooling Level')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.log_dir, f'training_progress_ep{episode}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary statistics.
        
        Returns:
            Dictionary of summary statistics
        """
        return {
            'total_episodes': len(self.episode_rewards),
            'final_avg_reward': np.mean(self.episode_rewards[-100:]),
            'final_avg_temp': np.mean(self.episode_avg_temps[-100:]),
            'total_violations': np.sum(self.episode_violations),
            'final_epsilon': self.agent.epsilon,
            'best_reward': np.max(self.episode_rewards),
            'best_episode': int(np.argmax(self.episode_rewards))
        }
    
    def evaluate(
        self,
        num_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render environment
            
        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_temps = []
        eval_violations = []
        
        for _ in tqdm(range(num_episodes), desc="Evaluating"):
            state, _ = self.env.reset()
            episode_reward = 0.0
            total_temp = 0.0
            total_violations = 0
            steps = 0
            done = False
            
            while not done:
                # Greedy action selection (no exploration)
                action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                total_temp += info['avg_temperature']
                total_violations += info['hotspots']
                steps += 1
                
                if render:
                    self.env.render()
            
            eval_rewards.append(episode_reward)
            eval_temps.append(total_temp / steps if steps > 0 else 0)
            eval_violations.append(total_violations)
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_temperature': np.mean(eval_temps),
            'total_violations': np.sum(eval_violations),
            'success_rate': np.sum(np.array(eval_violations) == 0) / num_episodes
        }
