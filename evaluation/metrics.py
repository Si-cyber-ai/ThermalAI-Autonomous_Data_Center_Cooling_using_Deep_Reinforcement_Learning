"""
Performance Metrics for Data Center Cooling Evaluation

Defines metrics for evaluating cooling controller performance.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import pandas as pd


class CoolingMetrics:
    """
    Comprehensive metrics for cooling system evaluation.
    """
    
    @staticmethod
    def compute_energy_consumption(
        cooling_history: List[np.ndarray],
        timestep: float = 60.0
    ) -> Dict[str, float]:
        """
        Compute energy consumption metrics.
        
        Args:
            cooling_history: List of cooling level arrays over time
            timestep: Simulation timestep in seconds
            
        Returns:
            Energy metrics dictionary
        """
        cooling_array = np.array(cooling_history)
        
        # Average power consumption (proportional to cooling level)
        avg_cooling = np.mean(cooling_array)
        
        # Peak power consumption
        peak_cooling = np.max(cooling_array)
        
        # Total energy (integrated cooling over time)
        total_energy = np.sum(cooling_array) * timestep / 3600.0  # Convert to kWh equivalent
        
        # Cooling variability (efficiency indicator)
        cooling_std = np.std(cooling_array)
        
        return {
            'avg_cooling_level': float(avg_cooling),
            'peak_cooling_level': float(peak_cooling),
            'total_energy': float(total_energy),
            'cooling_std': float(cooling_std),
            'energy_per_step': float(total_energy / len(cooling_history)) if len(cooling_history) > 0 else 0.0
        }
    
    @staticmethod
    def compute_temperature_metrics(
        temperature_history: List[np.ndarray],
        target_temp: float = 65.0,
        max_temp: float = 80.0
    ) -> Dict[str, float]:
        """
        Compute temperature-related metrics.
        
        Args:
            temperature_history: List of temperature arrays over time
            target_temp: Target operating temperature
            max_temp: Maximum safe temperature
            
        Returns:
            Temperature metrics dictionary
        """
        temp_array = np.array(temperature_history)
        
        # Average temperature
        avg_temp = np.mean(temp_array)
        
        # Maximum temperature reached
        max_temp_reached = np.max(temp_array)
        
        # Minimum temperature
        min_temp_reached = np.min(temp_array)
        
        # Temperature deviation from target
        temp_deviation = np.mean(np.abs(temp_array - target_temp))
        
        # Temperature stability (lower is better)
        temp_std = np.std(temp_array)
        
        # Violations count
        violations = np.sum(temp_array > max_temp)
        violation_ratio = violations / temp_array.size if temp_array.size > 0 else 0.0
        
        # Time in comfort zone (within ±5°C of target)
        comfort_zone_mask = np.abs(temp_array - target_temp) <= 5.0
        comfort_zone_ratio = np.mean(comfort_zone_mask)
        
        return {
            'avg_temperature': float(avg_temp),
            'max_temperature': float(max_temp_reached),
            'min_temperature': float(min_temp_reached),
            'temp_deviation': float(temp_deviation),
            'temp_std': float(temp_std),
            'violations': int(violations),
            'violation_ratio': float(violation_ratio),
            'comfort_zone_ratio': float(comfort_zone_ratio)
        }
    
    @staticmethod
    def compute_stability_metrics(
        temperature_history: List[np.ndarray],
        cooling_history: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute system stability metrics.
        
        Args:
            temperature_history: List of temperature arrays over time
            cooling_history: List of cooling arrays over time
            
        Returns:
            Stability metrics dictionary
        """
        temp_array = np.array(temperature_history)
        cooling_array = np.array(cooling_history)
        
        # Temperature oscillation (change between consecutive steps)
        temp_changes = np.abs(np.diff(temp_array, axis=0))
        avg_temp_change = np.mean(temp_changes)
        max_temp_change = np.max(temp_changes)
        
        # Cooling oscillation
        cooling_changes = np.abs(np.diff(cooling_array, axis=0))
        avg_cooling_change = np.mean(cooling_changes)
        max_cooling_change = np.max(cooling_changes)
        
        # Settling time (steps to reach stable state)
        # Defined as when temperature changes drop below threshold
        threshold = 0.5  # degrees
        settling_time = 0
        for i, change in enumerate(np.mean(temp_changes, axis=(1, 2))):
            if change < threshold:
                settling_time = i
                break
        
        return {
            'avg_temp_change': float(avg_temp_change),
            'max_temp_change': float(max_temp_change),
            'avg_cooling_change': float(avg_cooling_change),
            'max_cooling_change': float(max_cooling_change),
            'settling_time': int(settling_time)
        }
    
    @staticmethod
    def compute_responsiveness_metrics(
        temperature_history: List[np.ndarray],
        workload_history: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute controller responsiveness to workload changes.
        
        Args:
            temperature_history: List of temperature arrays over time
            workload_history: List of workload arrays over time
            
        Returns:
            Responsiveness metrics dictionary
        """
        temp_array = np.array(temperature_history)
        workload_array = np.array(workload_history)
        
        # Detect workload spikes
        workload_changes = np.diff(np.mean(workload_array, axis=(1, 2)))
        spike_threshold = 0.1
        spike_indices = np.where(workload_changes > spike_threshold)[0]
        
        if len(spike_indices) == 0:
            return {
                'avg_response_time': 0.0,
                'temp_overshoot': 0.0,
                'num_spikes_detected': 0
            }
        
        # Measure response time (steps to stabilize after spike)
        response_times = []
        overshoots = []
        
        for spike_idx in spike_indices:
            if spike_idx + 20 < len(temp_array):
                # Temperature trajectory after spike
                post_spike_temps = np.mean(temp_array[spike_idx:spike_idx+20], axis=(1, 2))
                
                # Find when temperature stabilizes
                baseline = post_spike_temps[0]
                for i, temp in enumerate(post_spike_temps[1:]):
                    if abs(temp - baseline) < 1.0:
                        response_times.append(i)
                        break
                
                # Measure overshoot
                overshoot = np.max(post_spike_temps) - baseline
                overshoots.append(overshoot)
        
        return {
            'avg_response_time': float(np.mean(response_times)) if response_times else 0.0,
            'temp_overshoot': float(np.mean(overshoots)) if overshoots else 0.0,
            'num_spikes_detected': len(spike_indices)
        }
    
    @staticmethod
    def compute_hotspot_metrics(
        temperature_history: List[np.ndarray],
        threshold: float = 75.0
    ) -> Dict[str, float]:
        """
        Compute hotspot formation metrics.
        
        Args:
            temperature_history: List of temperature arrays over time
            threshold: Temperature threshold for hotspot
            
        Returns:
            Hotspot metrics dictionary
        """
        temp_array = np.array(temperature_history)
        
        # Count hotspots over time
        hotspot_counts = np.sum(temp_array > threshold, axis=(1, 2))
        
        # Maximum simultaneous hotspots
        max_hotspots = np.max(hotspot_counts)
        
        # Average hotspot count
        avg_hotspots = np.mean(hotspot_counts)
        
        # Hotspot persistence (steps with at least one hotspot)
        hotspot_steps = np.sum(hotspot_counts > 0)
        hotspot_ratio = hotspot_steps / len(temperature_history) if len(temperature_history) > 0 else 0.0
        
        # Spatial hotspot concentration
        total_hotspots = np.sum(temp_array > threshold)
        unique_hotspot_locations = np.sum(np.any(temp_array > threshold, axis=0))
        concentration = total_hotspots / max(unique_hotspot_locations, 1)
        
        return {
            'max_hotspots': int(max_hotspots),
            'avg_hotspots': float(avg_hotspots),
            'hotspot_ratio': float(hotspot_ratio),
            'hotspot_concentration': float(concentration)
        }
    
    @staticmethod
    def compute_energy_saved(
        rl_cooling_history: List[np.ndarray],
        baseline_cooling_history: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute energy saved by RL controller compared to a baseline.

        Uses average energy per timestep (not totals) so that runs of
        different lengths are comparable.  Both histories are trimmed
        to the shorter length so the comparison spans the same number
        of steps.

        Formula:
            EnergySaved% = ((PID_avg - RL_avg) / PID_avg) * 100

        Returns:
            Dictionary with rl_avg_energy, baseline_avg_energy, steps,
            and energy_saved_percent (clamped to [-100, +100]).
        """
        # Equalise step counts
        n = min(len(rl_cooling_history), len(baseline_cooling_history))
        if n == 0:
            return {
                "rl_avg_energy": 0.0,
                "baseline_avg_energy": 0.0,
                "steps": 0,
                "energy_saved_percent": 0.0,
            }

        rl_total = float(sum(np.mean(c) for c in rl_cooling_history[:n]))
        bl_total = float(sum(np.mean(c) for c in baseline_cooling_history[:n]))

        rl_avg = rl_total / n
        bl_avg = bl_total / n

        if bl_avg > 0:
            saved_pct = ((bl_avg - rl_avg) / bl_avg) * 100
            saved_pct = max(min(saved_pct, 100.0), -100.0)
        else:
            saved_pct = 0.0

        return {
            "rl_avg_energy": round(rl_avg, 6),
            "baseline_avg_energy": round(bl_avg, 6),
            "steps": n,
            "energy_saved_percent": round(saved_pct, 4),
        }

    @staticmethod
    def compute_comprehensive_metrics(
        temperature_history: List[np.ndarray],
        cooling_history: List[np.ndarray],
        workload_history: List[np.ndarray],
        target_temp: float = 65.0,
        max_temp: float = 80.0,
        timestep: float = 60.0
    ) -> Dict[str, Any]:
        """
        Compute all metrics together.
        
        Args:
            temperature_history: Temperature history
            cooling_history: Cooling history
            workload_history: Workload history
            target_temp: Target temperature
            max_temp: Maximum safe temperature
            timestep: Simulation timestep
            
        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {}
        
        # Energy metrics
        metrics['energy'] = CoolingMetrics.compute_energy_consumption(
            cooling_history, timestep
        )
        
        # Temperature metrics
        metrics['temperature'] = CoolingMetrics.compute_temperature_metrics(
            temperature_history, target_temp, max_temp
        )
        
        # Stability metrics
        metrics['stability'] = CoolingMetrics.compute_stability_metrics(
            temperature_history, cooling_history
        )
        
        # Responsiveness metrics
        metrics['responsiveness'] = CoolingMetrics.compute_responsiveness_metrics(
            temperature_history, workload_history
        )
        
        # Hotspot metrics
        metrics['hotspots'] = CoolingMetrics.compute_hotspot_metrics(
            temperature_history, max_temp - 5.0
        )
        
        return metrics


def compare_controllers(
    rl_metrics: Dict[str, Any],
    pid_metrics: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create comparison table between RL and PID controllers.
    
    Args:
        rl_metrics: Metrics from RL controller
        pid_metrics: Metrics from PID controller
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    # Flatten metrics for comparison
    def flatten_metrics(metrics, prefix=''):
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                flat.update(flatten_metrics(value, f"{prefix}{key}_"))
            else:
                flat[f"{prefix}{key}"] = value
        return flat
    
    rl_flat = flatten_metrics(rl_metrics)
    pid_flat = flatten_metrics(pid_metrics)
    
    # Create comparison
    for metric_name in rl_flat.keys():
        if metric_name in pid_flat:
            rl_val = rl_flat[metric_name]
            pid_val = pid_flat[metric_name]
            
            # Compute improvement
            if isinstance(rl_val, (int, float)) and isinstance(pid_val, (int, float)):
                if pid_val != 0:
                    improvement = ((pid_val - rl_val) / abs(pid_val)) * 100
                else:
                    improvement = 0.0
            else:
                improvement = None
            
            comparison_data.append({
                'Metric': metric_name,
                'RL': rl_val,
                'PID': pid_val,
                'Improvement (%)': improvement
            })
    
    df = pd.DataFrame(comparison_data)
    return df
