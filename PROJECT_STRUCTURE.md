# Project Structure

> AI-Based Data Center Cooling Optimization using Safe RL and Digital Twin Simulation

```
Data Centers/
│
├── agents/                        # Supervisory AI agent
│   ├── __init__.py
│   └── cooling_agent.py           # CoolingAgent – supervisor that selects strategy based on risk level
│
├── checkpoints/                   # Saved model weights (auto-generated)
│   ├── dqn_episode_*.pth          # Periodic checkpoints during training
│   └── dqn_final.pth              # Final trained model
│
├── controllers/                   # Low-level controllers
│   ├── __init__.py
│   └── pid_controller.py          # PID controller (baseline comparator)
│
├── evaluation/                    # Experiment orchestration & metrics
│   ├── __init__.py
│   ├── experiments.py             # ExperimentRunner – runs RL vs PID comparisons
│   └── metrics.py                 # CoolingMetrics – energy saved %, temperature stats
│
├── frontend/                      # Streamlit dashboard
│   ├── __init__.py
│   └── dashboard.py               # 7-page interactive dashboard (Digital Twin, Training
│                                  #   Monitor, Action Distribution, Temperature Heatmap,
│                                  #   RL vs PID Comparison, Training Performance,
│                                  #   Real System Monitor)
│
├── logs/                          # Training artefacts (auto-generated)
│   ├── episode_logs.csv           # Per-episode reward/temperature/energy logs
│   ├── training_logs.csv          # Step-level training metrics
│   └── training_progress_ep*.png  # Progress plots saved during training
│
├── monitoring/                    # System monitoring utilities
│   ├── __init__.py
│   ├── laptop_sensors.py          # LaptopSensorMonitor – reads CPU/memory via psutil
│   └── training_logger.py         # TrainingLogger – CSV & plot logging during training
│
├── rl_agent/                      # Reinforcement learning agent
│   ├── __init__.py
│   ├── dqn_agent.py               # DQN agent (256-256-128 MLP, epsilon-greedy)
│   └── training_pipeline.py       # Training loop with shaped reward & entropy bonus
│
├── safety/                        # Safety systems
│   ├── __init__.py
│   ├── safety_filter.py           # SafetyFilter – tiered escalation (warning → critical → emergency)
│   └── safety_override.py         # SafetyOverride – legacy threshold-based override
│
├── simulator/                     # Digital Twin simulation environment
│   ├── __init__.py
│   ├── heat_transfer_model.py     # Physics-based heat diffusion model (3×4 grid)
│   └── thermal_environment.py     # Gymnasium environment with tiered safety escalation
│
├── tests/                         # Automated test suites
│   ├── __init__.py
│   ├── run_full_validation.py     # Combined test runner (runs all suites, prints report)
│   ├── run_scenarios.py           # 6-scenario integration tests (normal/spike/gradual/mixed/extreme/sustained)
│   ├── test_agent_behavior.py     # 4 agent strategy tests (normal/warning/critical/emergency)
│   └── test_energy_metrics.py     # Energy metric validation tests
│
├── workload/                      # Workload generation
│   ├── __init__.py
│   ├── dataset_loader.py          # WorkloadTraceLoader – loads real workload CSVs
│   └── synthetic_generator.py     # SyntheticWorkloadGenerator – creates artificial patterns
│
├── config.yaml                    # Central configuration (grid size, thresholds, hyperparameters)
├── train_model.py                 # CLI entry point – train the DQN agent
├── run_simulation.py              # CLI entry point – run RL vs PID experiment
├── validate_fixes.py              # CLI entry point – 5-run validation script
├── generate_research_graphs.py    # Generates 8 publication-quality figures (PNG + PDF)
├── run_dashboard.bat              # Windows batch launcher for Streamlit
├── run_dashboard.ps1              # PowerShell launcher for Streamlit
├── requirements.txt               # Python dependencies
│
├── graphs/                        # Auto-generated research figures (PNG + PDF)
│   ├── 01_temperature_comparison.*
│   ├── 02_energy_comparison.*
│   ├── 03_training_reward_curve.*
│   ├── 04_temperature_stability.*
│   ├── 05_action_distribution.*
│   ├── 06_rack_temperature_heatmap.*
│   ├── 07_workload_spike_response.*
│   └── 08_long_term_stability.*
│
├── README.md                      # Full project documentation
├── QUICKSTART.md                  # Quick-start guide
├── QUICK_REFERENCE.txt            # Command cheat-sheet
├── DASHBOARD_GUIDE.md             # Dashboard usage guide
├── MODEL_DETAILS.md               # Model architecture & pipeline documentation
├── MODELS_USED.md                 # Models & algorithms documentation
├── TRAINING_DETAILS.md            # Training hyperparameters & results
└── PROJECT_STRUCTURE.md           # This file
```

## Architecture

```
Workload Generator ──► Digital Twin (Gymnasium Env)
                             │
                             ▼
                      CoolingAgent (Supervisor)
                       ┌─────┴─────┐
                       │           │
                    DQN Agent   PID Controller
                       └─────┬─────┘
                             ▼
                       SafetyFilter (tiered escalation)
                             │
                             ▼
                      Environment Step
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Grid size | 3 × 4 (12 server racks) |
| Target temperature | 60 °C |
| Warning threshold | 70 °C |
| Hotspot threshold | 75 °C |
| Critical threshold | 80 °C |
| Emergency threshold | 85 °C |
| Thermal coupling (α) | 0.12 |
| Cooling responsiveness (β) | 0.30 |
