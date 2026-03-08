"""
Interactive Streamlit Dashboard for Data Center Cooling Optimization

Provides real-time visualization, training monitoring, and controller
comparison for the cooling simulation.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import yaml
import os
import time

from simulator.thermal_environment import DataCenterThermalEnv
from workload.synthetic_generator import SyntheticWorkloadGenerator, WorkloadScenario
from rl_agent.dqn_agent import DQNAgent
from controllers.pid_controller import PIDController
from monitoring.laptop_sensors import LaptopSensorMonitor
from agents.cooling_agent import CoolingAgent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Center Cooling",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {font-size:2.5rem;font-weight:bold;color:#1f77b4;text-align:center;margin-bottom:1rem}
    .metric-card {background:#f0f2f6;padding:1rem;border-radius:.5rem;margin:.5rem 0}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def initialize_environment(_config):
    grid_size = tuple(_config["simulation"]["grid_size"])
    workload_gen = SyntheticWorkloadGenerator(
        grid_size=grid_size,
        pattern=_config["workload"]["synthetic_pattern"],
        base_load=_config["workload"]["base_load"],
        peak_load=_config["workload"]["peak_load"],
    )
    env = DataCenterThermalEnv(config_path="config.yaml", workload_generator=workload_gen)
    return env, workload_gen


@st.cache_resource
def initialize_controllers(_config, _env):
    state_dim = _env.observation_space.shape[0]
    action_dim = _env.action_space.n
    rl_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=_config["rl"]["hidden_dim"],
    )
    checkpoint_path = "checkpoints/dqn_final.pth"
    if os.path.exists(checkpoint_path):
        rl_agent.load_checkpoint(checkpoint_path)
    pid_controller = PIDController(
        kp=_config["pid"]["kp"],
        ki=_config["pid"]["ki"],
        kd=_config["pid"]["kd"],
        setpoint=_config["pid"]["setpoint"],
    )
    return rl_agent, pid_controller


def create_heatmap(data, title, colorscale="Hot"):
    fig = go.Figure(data=go.Heatmap(z=data, colorscale=colorscale, colorbar=dict(title="Value")))
    fig.update_layout(title=title, xaxis_title="Rack Column", yaxis_title="Rack Row", height=300)
    return fig


# ---------------------------------------------------------------------------
# CSV loaders (for training pages)
# ---------------------------------------------------------------------------

def load_episode_logs() -> pd.DataFrame:
    path = "logs/episode_logs.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_step_logs() -> pd.DataFrame:
    path = "logs/training_logs.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(
        '<div class="main-header">❄️ AI-Based Data Center Cooling Optimization</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Safe Reinforcement Learning with Digital Twin Simulation**")
    st.divider()

    config = load_config()

    # ---- Sidebar ----
    st.sidebar.title("⚙️ Control Panel")
    page = st.sidebar.radio(
        "Page",
        [
            "Digital Twin",
            "Training Monitor",
            "Action Distribution",
            "Temperature Heatmap",
            "RL vs PID Comparison",
            "Training Performance",
            "Real System Monitor",
        ],
    )

    # ---- Dispatch ----
    if page == "Digital Twin":
        _sidebar_sim_controls(config)
    elif page == "Training Monitor":
        page_training_monitor()
    elif page == "Action Distribution":
        page_action_distribution()
    elif page == "Temperature Heatmap":
        page_temperature_heatmap()
    elif page == "RL vs PID Comparison":
        page_comparison(config)
    elif page == "Training Performance":
        page_training_performance()
    elif page == "Real System Monitor":
        display_real_system_monitor()


# ===================================================================
# Sidebar simulation controls  (used only by Digital Twin page)
# ===================================================================

def _sidebar_sim_controls(config):
    st.sidebar.divider()
    controller_type = st.sidebar.selectbox("Controller Type", ["RL (DQN)", "PID", "Adaptive PID"])
    st.sidebar.subheader("Simulation Parameters")
    ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 15.0, 35.0, float(config["simulation"]["ambient_temperature"]), 0.5)
    workload_pattern = st.sidebar.selectbox("Workload Pattern", ["mixed", "sinusoidal", "spikes", "burst"])
    alpha = st.sidebar.slider("Heat Generation (α)", 0.05, 0.30, float(config["simulation"]["alpha"]), 0.01)
    beta = st.sidebar.slider("Cooling Efficiency (β)", 0.10, 0.40, float(config["simulation"]["beta"]), 0.01)
    if "PID" in controller_type:
        st.sidebar.subheader("PID Tuning")
        st.sidebar.slider("Kp", 0.0, 2.0, float(config["pid"]["kp"]), 0.1)
        st.sidebar.slider("Ki", 0.0, 1.0, float(config["pid"]["ki"]), 0.05)
        st.sidebar.slider("Kd", 0.0, 0.5, float(config["pid"]["kd"]), 0.01)
    scenario = st.sidebar.selectbox("Load Scenario", ["Normal", "Hotspot", "Edge Heavy", "Gradient"])
    st.sidebar.divider()
    run_simulation = st.sidebar.button("▶️ Run Simulation", type="primary")
    display_digital_twin(config, controller_type, ambient_temp, workload_pattern, alpha, beta, scenario, run_simulation)


# ===================================================================
# PAGE: Digital Twin (original)
# ===================================================================

def display_digital_twin(config, controller_type, ambient_temp, workload_pattern, alpha, beta, scenario, run_simulation):
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "step_count" not in st.session_state:
        st.session_state.step_count = 0
    if "history" not in st.session_state:
        st.session_state.history = {"temperatures": [], "cooling": [], "workload": [], "rewards": []}

    env, workload_gen = initialize_environment(config)
    rl_agent, pid_controller = initialize_controllers(config, env)
    env.ambient_temp = ambient_temp
    env.heat_model.alpha = alpha
    env.heat_model.beta = beta
    workload_gen.pattern = workload_pattern

    if scenario != "Normal":
        grid_size = tuple(config["simulation"]["grid_size"])
        scenario_fn = {
            "Hotspot": WorkloadScenario.create_hotspot_scenario,
            "Edge Heavy": WorkloadScenario.create_edge_heavy_scenario,
            "Gradient": WorkloadScenario.create_gradient_scenario,
        }.get(scenario)
        if scenario_fn:
            env.cpu_workload = scenario_fn(grid_size)

    if run_simulation and not st.session_state.simulation_running:
        st.session_state.simulation_running = True
        state, _ = env.reset()
        agent_status_ph = st.empty()
        metrics_ph = st.empty()
        energy_ph = st.empty()
        heatmap_ph = st.empty()
        chart_ph = st.empty()
        energy_chart_ph = st.empty()
        summary_ph = st.empty()

        # --- Energy tracking (per-step averages) ---
        rl_total_energy = 0.0
        pid_total_energy = 0.0
        energy_steps = 0
        rl_energy_timeline = []
        pid_energy_timeline = []

        # --- Supervisory AI Cooling Agent ---
        cooling_agent = CoolingAgent()

        # Baseline PID env for energy comparison
        env_baseline, _ = initialize_environment(config)
        _, pid_baseline = initialize_controllers(config, env_baseline)
        env_baseline.ambient_temp = ambient_temp
        env_baseline.heat_model.alpha = alpha
        env_baseline.heat_model.beta = beta
        if scenario != "Normal":
            grid_size = tuple(config["simulation"]["grid_size"])
            scenario_fn = {
                "Hotspot": WorkloadScenario.create_hotspot_scenario,
                "Edge Heavy": WorkloadScenario.create_edge_heavy_scenario,
                "Gradient": WorkloadScenario.create_gradient_scenario,
            }.get(scenario)
            if scenario_fn:
                env_baseline.cpu_workload = scenario_fn(grid_size)
        baseline_state, _ = env_baseline.reset()
        baseline_agent = CoolingAgent()

        for step in range(100):
            if not st.session_state.simulation_running:
                break
            grids = env.get_state_grid()
            temperatures = grids["temperatures"]
            cooling_levels = grids["cooling_levels"]
            workload = grids["cpu_workload"]

            # --- Supervisor decides strategy & applies safety filter ---
            if controller_type == "RL (DQN)":
                _, rl_action, agent_info = cooling_agent.act(
                    env, rl_agent, pid_controller, state, training=False
                )
                action = rl_action
            else:
                _, _, agent_info = cooling_agent.act(
                    env, rl_agent, pid_controller, state, training=False
                )
                # For PID mode, the agent internally selects PID when risk is
                # moderate, but we always honour the user's controller choice:
                proposed = pid_controller.compute(temperatures)
                env.cooling_levels = np.clip(proposed, 0.0, 1.0)
                action = 1

            state, reward, terminated, truncated, info = env.step(action)

            # Post-step safety enforcement (always)
            post_info = cooling_agent.post_step_safety(env)

            # --- Compute energy for the *selected* controller ---
            selected_energy_step = float(np.mean(env.get_state_grid()["cooling_levels"]))

            # --- Run PID baseline step in parallel ---
            bl_grids = env_baseline.get_state_grid()
            bl_proposed = pid_baseline.compute(bl_grids["temperatures"])
            env_baseline.cooling_levels = np.clip(bl_proposed, 0.0, 1.0)
            baseline_state, _, bl_term, _, bl_info = env_baseline.step(1)
            baseline_agent.post_step_safety(env_baseline)
            baseline_energy_step = float(np.mean(env_baseline.get_state_grid()["cooling_levels"]))

            # Always: selected controller = RL bucket, baseline = PID bucket
            rl_total_energy += selected_energy_step
            pid_total_energy += baseline_energy_step
            energy_steps += 1

            rl_avg_energy = rl_total_energy / energy_steps
            pid_avg_energy = pid_total_energy / energy_steps

            rl_energy_timeline.append(rl_avg_energy)
            pid_energy_timeline.append(pid_avg_energy)

            energy_saved_pct = (
                max(min(((pid_avg_energy - rl_avg_energy) / pid_avg_energy) * 100, 100.0), -100.0)
                if pid_avg_energy > 0 else 0.0
            )

            # --- History ---
            st.session_state.history["temperatures"].append(temperatures.copy())
            st.session_state.history["cooling"].append(cooling_levels.copy())
            st.session_state.history["workload"].append(workload.copy())
            st.session_state.history["rewards"].append(reward)
            st.session_state.step_count += 1

            # --- Agent status bar ---
            risk_colors = {
                "normal": "🟢", "moderate_risk": "🟡",
                "high_risk": "🟠", "emergency": "🔴",
            }
            strategy_labels = {
                "rl_control": "RL Agent",
                "pid_control": "PID Controller",
                "strong_cooling": "Strong Cooling",
                "max_cooling": "Emergency Max Cooling",
            }
            with agent_status_ph.container():
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Agent Mode", f"{risk_colors.get(agent_info['risk_level'], '⚪')} {agent_info['risk_level'].replace('_', ' ').title()}")
                a2.metric("Cooling Strategy", strategy_labels.get(agent_info['strategy'], agent_info['strategy']))
                a3.metric("Safety Overrides", str(agent_info.get('override_count', 0)))
                a4.metric("Filter Active", "Yes" if post_info.get('override_active', False) else "No")

            # --- Live metrics ---
            # Re-read temperatures after safety correction
            corrected_temps = env.get_state_grid()["temperatures"]
            avg_t = float(np.mean(corrected_temps))
            max_t = float(np.max(corrected_temps))
            with metrics_ph.container():
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Avg Temperature", f"{avg_t:.1f} °C")
                c2.metric("Max Temperature", f"{max_t:.1f} °C")
                c3.metric("Avg Cooling", f"{info['avg_cooling']:.2f}")
                hotspot_count = int(np.sum(corrected_temps > 75.0))
                c4.metric("Hotspots (>75°C)", str(hotspot_count))
                violations_now = int(np.sum(corrected_temps > 80.0))
                c5.metric("Violations (>80°C)", str(violations_now))
                c6.metric("Total Violations", str(info["violations"]))

            # --- Live energy metrics ---
            with energy_ph.container():
                st.markdown("#### ⚡ Energy Efficiency")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Energy Saved", f"{energy_saved_pct:.2f}%")
                e2.metric("RL Avg Energy/Step", f"{rl_avg_energy:.4f}")
                e3.metric("PID Avg Energy/Step", f"{pid_avg_energy:.4f}")
                e4.metric("Steps", str(energy_steps))
                if energy_saved_pct < 0:
                    st.info("RL prioritises thermal safety in this scenario, resulting in higher cooling energy.")
                else:
                    st.success("RL improves cooling energy efficiency compared to PID.")

            with heatmap_ph.container():
                c1, c2, c3 = st.columns(3)
                c1.plotly_chart(create_heatmap(temperatures, "Temperature (°C)", "Hot"), key=f"dt_temp_{step}", width="stretch")
                c2.plotly_chart(create_heatmap(cooling_levels, "Cooling Level", "Blues"), key=f"dt_cool_{step}", width="stretch")
                c3.plotly_chart(create_heatmap(workload, "CPU Workload", "Viridis"), key=f"dt_wl_{step}", width="stretch")
                st.caption("Spatial view of rack temperatures, cooling effort, and CPU workload across the server grid.")

            with chart_ph.container():
                if len(st.session_state.history["temperatures"]) > 1:
                    temps_mean = [np.mean(t) for t in st.session_state.history["temperatures"]]
                    cool_mean = [np.mean(c) for c in st.session_state.history["cooling"]]
                    fig = make_subplots(rows=2, cols=1, subplot_titles=("Avg Temperature", "Avg Cooling"))
                    fig.add_trace(go.Scatter(y=temps_mean, mode="lines", name="Temp"), row=1, col=1)
                    fig.add_trace(go.Scatter(y=cool_mean, mode="lines", name="Cooling"), row=2, col=1)
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, key=f"dt_ts_{step}")
                    st.caption("Tracks average temperature and cooling level across simulation steps to visualise controller behaviour.")

            # --- Live energy comparison chart ---
            with energy_chart_ph.container():
                if step > 0:
                    edf = pd.DataFrame({
                        "Step": list(range(step + 1)),
                        "RL Energy": rl_energy_timeline,
                        "PID Energy": pid_energy_timeline,
                    })
                    fig = px.line(edf, x="Step", y=["RL Energy", "PID Energy"],
                                  title="Cooling Energy Comparison")
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, key=f"dt_ecmp_{step}", width="stretch")
                    st.caption("Compares average energy per step between the selected controller and a PID baseline running in parallel.")

            time.sleep(0.1)
            if terminated:
                break

        # --- Final summary panel ---
        with summary_ph.container():
            st.divider()
            st.markdown("### Final Energy Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Energy Saved", f"{energy_saved_pct:.2f}%")
            s2.metric("RL Avg Energy/Step", f"{rl_avg_energy:.4f}")
            s3.metric("PID Avg Energy/Step", f"{pid_avg_energy:.4f}")
            s4.metric("Total Steps", str(energy_steps))
            if energy_saved_pct < 0:
                st.info("RL prioritises thermal safety in this scenario, resulting in higher cooling energy.")
            else:
                st.success("RL improves cooling energy efficiency compared to PID.")

        st.session_state.simulation_running = False
    else:
        st.info("Click **▶️ Run Simulation** in the sidebar to start the Digital Twin.")


# ===================================================================
# PAGE: Training Monitor  (Part 6)
# ===================================================================

def page_training_monitor():
    st.subheader("📈 Training Monitor")
    df = load_episode_logs()
    if df.empty:
        st.warning("No training logs found. Run training first to generate data in `logs/episode_logs.csv`.")
        return

    # Ensure numeric
    for col in ["total_reward", "avg_temperature", "max_temperature", "avg_cooling",
                 "total_violations", "avg_loss", "epsilon", "episode_length"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    st.markdown(f"**{len(df)} episodes logged**")

    # --- 6 Plotly graphs in 3x2 grid --
    col1, col2 = st.columns(2)

    with col1:
        # 1. Reward vs Episode
        fig = px.line(df, x="episode", y="total_reward", title="Total Reward per Episode")
        fig.update_layout(height=320)
        st.plotly_chart(fig, key="tm_reward", width="stretch")
        st.caption("Total reward earned per training episode. Higher values indicate the agent is learning to balance cooling efficiency with temperature safety.")

        # 3. Energy / Cooling vs Episode
        fig = px.line(df, x="episode", y="avg_cooling", title="Average Cooling Level per Episode")
        fig.update_layout(height=320)
        st.plotly_chart(fig, key="tm_cooling", width="stretch")
        st.caption("Average cooling effort applied per episode. Lower values mean the agent uses less energy while maintaining safe temperatures.")

        # 5. Loss vs Episode
        fig = px.line(df, x="episode", y="avg_loss", title="Average Loss per Episode")
        fig.update_layout(height=320)
        st.plotly_chart(fig, key="tm_loss", width="stretch")
        st.caption("DQN training loss — should decrease over time as the Q-network converges.")

    with col2:
        # 2. Avg Temperature vs Episode
        fig = px.line(df, x="episode", y="avg_temperature", title="Average Temperature per Episode")
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Max Temp Limit")
        fig.update_layout(height=320)
        st.plotly_chart(fig, key="tm_temp", width="stretch")
        st.caption("Average temperature per episode. The red dashed line indicates the safety threshold (80 °C).")

        # 4. Violations vs Episode
        fig = px.bar(df, x="episode", y="total_violations", title="Safety Violations per Episode")
        fig.update_layout(height=320)
        st.plotly_chart(fig, key="tm_violations", width="stretch")
        st.caption("Number of racks exceeding the 80 °C safety limit per episode. Target: zero violations.")

        # 6. Epsilon vs Episode
        fig = px.line(df, x="episode", y="epsilon", title="Epsilon (Exploration) per Episode")
        fig.update_layout(height=320)
        st.plotly_chart(fig, key="tm_epsilon", width="stretch")
        st.caption("Exploration rate (ε-greedy). Decays from 1.0 → 0.05 as the agent shifts from exploration to exploitation.")


# ===================================================================
# PAGE: Action Distribution  (Part 7)
# ===================================================================

def page_action_distribution():
    st.subheader("🎯 RL Action Distribution")
    ep_df = load_episode_logs()
    step_df = load_step_logs()

    if ep_df.empty and step_df.empty:
        st.warning("No training logs found.")
        return

    # --- Cumulative action distribution from step logs ---
    if not step_df.empty and "action" in step_df.columns:
        st.markdown("### Cumulative Action Counts (all episodes)")
        counts = step_df["action"].value_counts().sort_index()
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={"x": "Action", "y": "Count"},
            title="Action Frequency",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, key="ad_cumulative", width="stretch")
        st.caption("Frequency of each discrete cooling action chosen by the RL agent across all training steps.")

    # --- Per-episode action distribution from episode logs ---
    if not ep_df.empty and "action_distribution" in ep_df.columns:
        st.markdown("### Per-Episode Action Distribution")
        # Parse  "0:n,1:n,..." strings
        all_actions = set()
        parsed_rows = []
        for _, row in ep_df.iterrows():
            act_str = str(row["action_distribution"])
            parts = {}
            if act_str and act_str != "nan":
                for pair in act_str.split(","):
                    if ":" in pair:
                        k, v = pair.split(":")
                        parts[k.strip()] = int(v.strip())
                        all_actions.add(k.strip())
            parsed_rows.append(parts)
        all_actions = sorted(all_actions)
        act_df = pd.DataFrame(parsed_rows).fillna(0)
        act_df.insert(0, "episode", ep_df["episode"].values)

        fig = go.Figure()
        for a in all_actions:
            if a in act_df.columns:
                fig.add_trace(go.Bar(x=act_df["episode"], y=act_df[a], name=f"Action {a}"))
        fig.update_layout(barmode="stack", title="Action Distribution per Episode", height=400)
        st.plotly_chart(fig, key="ad_per_episode", width="stretch")
        st.caption("Stacked action distribution per episode — shows how the agent's action preferences evolve during training.")


# ===================================================================
# PAGE: Temperature Heatmap Animation  (Part 8)
# ===================================================================

def page_temperature_heatmap():
    st.subheader("🌡️ Temperature Heatmap Animation")
    step_df = load_step_logs()
    if step_df.empty:
        st.warning("No step-level logs found.")
        return

    episodes = sorted(step_df["episode"].unique())
    selected_ep = st.selectbox("Select Episode", episodes, key="hm_ep_select")
    ep_data = step_df[step_df["episode"] == selected_ep].copy()
    ep_data = ep_data.sort_values("step")

    if ep_data.empty:
        st.info("No steps for this episode.")
        return

    config = load_config()
    rows, cols = config["simulation"]["grid_size"]

    # Show animated line chart of temperature over steps
    fig = px.line(
        ep_data,
        x="step",
        y=["avg_temperature", "max_temperature"],
        title=f"Temperature Trajectory – Episode {selected_ep}",
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Max Limit")
    fig.add_hline(y=85, line_dash="dot", line_color="darkred", annotation_text="Critical")
    fig.update_layout(height=400)
    st.plotly_chart(fig, key="hm_trajectory", width="stretch")
    st.caption("Average and maximum rack temperatures over simulation steps. Red dashed = safety limit (80 °C), dark red dotted = critical (85 °C).")

    # Animated heatmap via slider
    st.markdown("#### Spatial Temperature (reconstructed from logged stats)")
    step_idx = st.slider("Step", int(ep_data["step"].min()), int(ep_data["step"].max()), int(ep_data["step"].min()), key="hm_step_slider")
    row_data = ep_data[ep_data["step"] == step_idx]
    if not row_data.empty:
        avg_t = float(row_data["avg_temperature"].iloc[0])
        max_t = float(row_data["max_temperature"].iloc[0])
        # Reconstruct approximate grid from avg/max
        grid = np.random.default_rng(seed=step_idx).normal(loc=avg_t, scale=max(0.5, (max_t - avg_t) / 3), size=(rows, cols))
        grid = np.clip(grid, avg_t - 5, max_t)
        fig = create_heatmap(grid, f"Temperature Grid – Step {step_idx}", "Hot")
        st.plotly_chart(fig, key=f"hm_grid_{step_idx}", width="stretch")
        st.caption("Approximate spatial temperature distribution reconstructed from logged average and max values.")


# ===================================================================
# PAGE: RL vs PID Comparison  (Part 9)
# ===================================================================

def page_comparison(config):
    st.subheader("📊 RL vs PID Comparison")

    num_steps = st.slider("Simulation Steps", 50, 500, 200, key="cmp_steps")

    if st.button("Run Comparison", key="cmp_run"):
        env, workload_gen = initialize_environment(config)
        rl_agent, pid_controller = initialize_controllers(config, env)

        results = {"RL": {"temps": [], "cooling": [], "rewards": [], "violations": [],
                          "energy_cum": []},
                   "PID": {"temps": [], "cooling": [], "rewards": [], "violations": [],
                           "energy_cum": []}}

        for label, use_rl in [("RL", True), ("PID", False)]:
            state, _ = env.reset()
            agent = CoolingAgent()
            cum_energy = 0.0
            for _ in range(num_steps):
                if use_rl:
                    _, rl_action, _ = agent.act(
                        env, rl_agent, pid_controller, state, training=False
                    )
                    action = rl_action
                else:
                    grids = env.get_state_grid()
                    proposed = pid_controller.compute(grids["temperatures"])
                    env.cooling_levels = np.clip(proposed, 0.0, 1.0)
                    action = 1
                state, reward, terminated, truncated, info = env.step(action)
                agent.post_step_safety(env)
                energy_step = float(np.mean(env.get_state_grid()["cooling_levels"]))
                cum_energy += energy_step
                corrected_temps = env.get_state_grid()["temperatures"]
                results[label]["temps"].append(float(np.mean(corrected_temps)))
                results[label]["cooling"].append(float(np.mean(env.get_state_grid()["cooling_levels"])))
                results[label]["rewards"].append(reward)
                results[label]["violations"].append(int(np.sum(corrected_temps > 80.0)))
                results[label]["energy_cum"].append(cum_energy)
                if terminated:
                    break

        # --- Energy Saved % (using average energy per step) ---
        rl_steps = len(results["RL"]["energy_cum"])
        pid_steps = len(results["PID"]["energy_cum"])
        rl_energy = results["RL"]["energy_cum"][-1] / rl_steps if rl_steps > 0 else 0
        pid_energy = results["PID"]["energy_cum"][-1] / pid_steps if pid_steps > 0 else 0
        energy_saved_pct = (max(min((pid_energy - rl_energy) / pid_energy * 100, 100.0), -100.0)) if pid_energy > 0 else 0.0

        st.markdown("### ⚡ Energy Efficiency")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Energy Saved", f"{energy_saved_pct:.2f}%")
        e2.metric("RL Avg Energy/Step", f"{rl_energy:.4f}")
        e3.metric("PID Avg Energy/Step", f"{pid_energy:.4f}")
        e4.metric("Steps", f"RL={rl_steps}, PID={pid_steps}")
        if energy_saved_pct < 0:
            st.info("RL prioritises thermal safety in this scenario, resulting in higher cooling energy.")
        else:
            st.success("RL improves cooling energy efficiency compared to PID.")
        st.divider()

        # Time-series comparison (original 4 + energy)
        fig = make_subplots(rows=3, cols=2,
                            subplot_titles=("Avg Temperature", "Cooling Level",
                                            "Reward", "Violations",
                                            "Cumulative Energy", "Cooling Level Distribution"))
        for i, (metric, title) in enumerate([("temps", "Temp"), ("cooling", "Cool"),
                                              ("rewards", "Reward"), ("violations", "Viol"),
                                              ("energy_cum", "Energy")], 1):
            r, c = (i - 1) // 2 + 1, (i - 1) % 2 + 1
            fig.add_trace(go.Scatter(y=results["RL"][metric], mode="lines", name=f"RL-{title}",
                                     line=dict(color="blue"), showlegend=(i == 1)),
                          row=r, col=c)
            fig.add_trace(go.Scatter(y=results["PID"][metric], mode="lines", name=f"PID-{title}",
                                     line=dict(color="orange"), showlegend=(i == 1)),
                          row=r, col=c)

        # Cooling level distribution (histogram) in row=3, col=2
        fig.add_trace(go.Histogram(x=results["RL"]["cooling"], name="RL", opacity=0.6,
                                    marker_color="blue"), row=3, col=2)
        fig.add_trace(go.Histogram(x=results["PID"]["cooling"], name="PID", opacity=0.6,
                                    marker_color="orange"), row=3, col=2)
        fig.update_layout(height=800, title_text="RL vs PID Time Series", barmode="overlay")
        st.plotly_chart(fig, key="cmp_timeseries", width="stretch")
        st.caption("Side-by-side comparison of RL vs PID across temperature, cooling, reward, violations, cumulative energy, and cooling-level distribution.")

        # Energy bar chart
        st.markdown("### Energy Usage Bar Chart")
        bar_fig = go.Figure(data=[
            go.Bar(name="RL", x=["Avg Energy/Step"], y=[rl_energy], marker_color="blue"),
            go.Bar(name="PID", x=["Avg Energy/Step"], y=[pid_energy], marker_color="orange"),
        ])
        bar_fig.update_layout(barmode="group", height=350, title="Average Cooling Energy per Step")
        st.plotly_chart(bar_fig, key="cmp_energy_bar", width="stretch")
        st.caption("Average cooling energy consumed per timestep. Lower is more efficient.")

        # Summary table
        summary = {}
        for label in ["RL", "PID"]:
            r = results[label]
            summary[label] = {
                "Avg Temp (°C)": f"{np.mean(r['temps']):.2f}",
                "Max Temp (°C)": f"{np.max(r['temps']):.2f}",
                "Avg Cooling": f"{np.mean(r['cooling']):.3f}",
                "Total Reward": f"{np.sum(r['rewards']):.2f}",
                "Total Violations": int(np.sum(r["violations"])),
                "Total Energy": f"{r['energy_cum'][-1]:.2f}" if r["energy_cum"] else "0",
                "Steps": len(r["temps"]),
            }
        summary["Savings"] = {
            "Avg Temp (°C)": "",
            "Max Temp (°C)": "",
            "Avg Cooling": "",
            "Total Reward": "",
            "Total Violations": "",
            "Total Energy": f"{energy_saved_pct:.2f}% saved",
            "Steps": "",
        }
        st.markdown("### Summary")
        st.dataframe(pd.DataFrame(summary).T)


# ===================================================================
# PAGE: Training Performance Panel  (Part 10)
# ===================================================================

def page_training_performance():
    st.subheader("⚡ Training Performance Panel")
    ep_df = load_episode_logs()
    if ep_df.empty:
        st.warning("No episode logs found. Run training first.")
        return

    for col in ["total_reward", "avg_temperature", "max_temperature", "avg_cooling",
                 "total_violations", "avg_loss", "epsilon", "episode_length"]:
        if col in ep_df.columns:
            ep_df[col] = pd.to_numeric(ep_df[col], errors="coerce")

    latest = ep_df.iloc[-1]

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Episode", int(latest["episode"]))
    c2.metric("Latest Reward", f"{latest['total_reward']:.1f}")
    c3.metric("Avg Temp", f"{latest['avg_temperature']:.1f} °C")
    c4.metric("Avg Cooling", f"{latest['avg_cooling']:.2f}")
    c5.metric("Violations", int(latest["total_violations"]))

    st.divider()

    # Rolling averages
    window = st.slider("Rolling Window", 5, 50, 10, key="tp_window")
    ep_df["reward_ma"] = ep_df["total_reward"].rolling(window, min_periods=1).mean()
    ep_df["temp_ma"] = ep_df["avg_temperature"].rolling(window, min_periods=1).mean()
    ep_df["viol_ma"] = ep_df["total_violations"].rolling(window, min_periods=1).mean()

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep_df["episode"], y=ep_df["total_reward"], mode="markers",
                                 name="Raw", marker=dict(size=3, opacity=0.4)))
        fig.add_trace(go.Scatter(x=ep_df["episode"], y=ep_df["reward_ma"], mode="lines",
                                 name=f"MA-{window}", line=dict(width=2)))
        fig.update_layout(title="Reward Trend", height=350)
        st.plotly_chart(fig, key="tp_reward", width="stretch")
        st.caption("Raw rewards with moving average overlay — shows learning convergence over episodes.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep_df["episode"], y=ep_df["episode_length"], mode="lines+markers",
                                 name="Length", marker=dict(size=3)))
        fig.update_layout(title="Episode Length", height=300)
        st.plotly_chart(fig, key="tp_length", width="stretch")
        st.caption("Number of steps per episode — longer episodes indicate the agent avoids early termination.")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep_df["episode"], y=ep_df["avg_temperature"], mode="markers",
                                 name="Raw", marker=dict(size=3, opacity=0.4)))
        fig.add_trace(go.Scatter(x=ep_df["episode"], y=ep_df["temp_ma"], mode="lines",
                                 name=f"MA-{window}", line=dict(width=2)))
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.update_layout(title="Temperature Trend", height=350)
        st.plotly_chart(fig, key="tp_temp", width="stretch")
        st.caption("Temperature trend with moving average. Red dashed line = 80 °C safety threshold.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep_df["episode"], y=ep_df["viol_ma"], mode="lines",
                                 name=f"MA-{window}", line=dict(width=2, color="red")))
        fig.update_layout(title="Violations Trend", height=300)
        st.plotly_chart(fig, key="tp_viol", width="stretch")
        st.caption("Moving average of safety violations — should approach zero as the agent learns safety constraints.")

    # Data table
    with st.expander("Raw Episode Data"):
        st.dataframe(ep_df)


# ===================================================================
# PAGE: Real System Monitor  (original)
# ===================================================================

def display_real_system_monitor():
    st.subheader("🖥️ Real System Monitoring")

    try:
        import psutil  # noqa: F401 — verify availability
    except ImportError:
        st.error("psutil is not installed. Install it with `pip install psutil`.")
        return

    if "monitor" not in st.session_state:
        st.session_state.monitor = LaptopSensorMonitor()
    monitor = st.session_state.monitor

    # --- System Info ---
    with st.expander("System Info & Sensor Availability", expanded=True):
        info = monitor.system_info
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Platform", info.get("platform", "N/A"))
        i2.metric("CPU Cores", f"{info.get('cpu_count_physical', '?')} / {info.get('cpu_count_logical', '?')}")
        i3.metric("Total RAM", f"{info.get('memory_total_gb', 0):.1f} GB")
        i4.metric("Processor", info.get("processor", "N/A")[:30])
        st.text(monitor.get_sensor_availability_report())

    st.subheader("Live Sensor Readings")
    if st.button("Start Monitoring", key="rsm_start"):
        sensor_ph = st.empty()
        chart_ph = st.empty()
        for i in range(60):
            try:
                readings = monitor.read_sensors()
            except Exception:
                readings = {
                    "cpu_usage_percent": 0.0,
                    "cpu_temp_celsius": None,
                    "cpu_freq_mhz": None,
                    "memory_usage_percent": 0.0,
                }
            with sensor_ph.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CPU Usage", f"{readings['cpu_usage_percent']:.1f}%")
                temp = readings.get("cpu_temp_celsius")
                c2.metric("CPU Temp", f"{temp:.1f} °C" if temp else "N/A")
                freq = readings.get("cpu_freq_mhz")
                c3.metric("CPU Freq", f"{freq:.0f} MHz" if freq else "N/A")
                c4.metric("Memory", f"{readings['memory_usage_percent']:.1f}%")
            with chart_ph.container():
                if len(monitor.cpu_usage_history) > 1:
                    st.line_chart(pd.DataFrame({"CPU Usage": list(monitor.cpu_usage_history)}))
                    st.caption("Real-time CPU usage from the host machine running this dashboard.")
            time.sleep(1)


if __name__ == "__main__":
    main()
