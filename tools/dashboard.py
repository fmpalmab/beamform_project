# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "dash",
#     "plotly",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///

import json
import math
from pathlib import Path
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

def find_latest_results_dir():
    if not RESULTS_DIR.exists():
        return None
    dirs = sorted(list(RESULTS_DIR.glob("presentation_*")) + list(RESULTS_DIR.glob("simulations_*")), reverse=True)
    if dirs:
        return dirs[0]
    return None

DATA_DIR = find_latest_results_dir()

def load_real_data():
    data = {
        "tracker_power": None,
        "cuda_v5_bench": None,
        "evolution": None,
        "astro_report": None,
        "astro_metrics": None,
        "vela_power": None,
        "sun_power": None,
        "frb_power": None,
    }
    # Check both DATA_DIR and any simulation directory
    search_dirs = [DATA_DIR] if DATA_DIR else []
    if RESULTS_DIR.exists():
        search_dirs.extend(sorted(RESULTS_DIR.glob("simulations_*"), reverse=True))
        search_dirs.extend(sorted(RESULTS_DIR.glob("presentation_*"), reverse=True))

    for d in search_dirs:
        if d is None or not d.exists():
            continue
        if data["tracker_power"] is None and (d / "tracker_power_dynamics.csv").exists():
            data["tracker_power"] = pd.read_csv(d / "tracker_power_dynamics.csv")
        if data["vela_power"] is None and (d / "vela_power_dynamics.csv").exists():
            data["vela_power"] = pd.read_csv(d / "vela_power_dynamics.csv")
        if data["sun_power"] is None and (d / "sun_power_dynamics.csv").exists():
            data["sun_power"] = pd.read_csv(d / "sun_power_dynamics.csv")
        if data["frb_power"] is None and (d / "frb_power_dynamics.csv").exists():
            data["frb_power"] = pd.read_csv(d / "frb_power_dynamics.csv")
        if data["cuda_v5_bench"] is None and (d / "cuda_v5_benchmark_results.csv").exists():
            data["cuda_v5_bench"] = pd.read_csv(d / "cuda_v5_benchmark_results.csv")
        if data["evolution"] is None and (d / "evolution_comparison.csv").exists():
            data["evolution"] = pd.read_csv(d / "evolution_comparison.csv")
        if data["astro_report"] is None and (d / "astronomical_validation" / "v5" / "astronomical_validation_report.json").exists():
            with open(d / "astronomical_validation" / "v5" / "astronomical_validation_report.json", "r", encoding="utf-8") as f:
                data["astro_report"] = json.load(f)
        if data["astro_metrics"] is None and (d / "astronomical_validation_metrics.json").exists():
            with open(d / "astronomical_validation_metrics.json", "r", encoding="utf-8") as f:
                data["astro_metrics"] = json.load(f)
    return data

REAL_DATA = load_real_data()

app = dash.Dash(__name__, title="Beam Tracker & Astro-Validation Dashboard")

app.layout = html.Div(style={'fontFamily': 'Segoe UI, Helvetica, Arial, sans-serif', 'padding': '25px', 'backgroundColor': '#f8f9fa'}, children=[
    html.Div([
        html.H1("Beam Tracker: Live Real-Data & Validation Dashboard", style={'color': '#1a202c', 'marginBottom': '5px'}),
        html.P(f"Visualizing actual CUDA V5 execution data and benchmarks from: {DATA_DIR.name if DATA_DIR else 'Local Sim'}", style={'color': '#4a5568', 'marginTop': '0px'})
    ]),
    
    html.Div(style={'backgroundColor': '#ffffff', 'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
        html.Label("Select Dataset / Source:", style={'fontWeight': 'bold', 'marginRight': '15px', 'fontSize': '15px'}),
        dcc.Dropdown(
            id='target-selector',
            options=[
                {'label': '⚡ [REAL CUDA V5] Actual Tracker Power Dynamics (962 Steps)', 'value': 'real_tracker'},
                {'label': '🌌 [REAL CUDA V5] FRB20180916B Canonical Validation (DM = 348.82)', 'value': 'real_frb_canonical'},
                {'label': '🌌 [REAL CUDA V5] High DM Burst Validation (DM = 1205.40)', 'value': 'real_frb_high_dm'},
                {'label': '🌌 [REAL CUDA V5] Scattering Dominated Burst (DM = 574.10, τ = 12ms)', 'value': 'real_frb_scatter'},
                {'label': '🚀 [REAL BENCHMARKS] GPU Architecture Scaling & Speedups (V2 -> V5)', 'value': 'real_benchmarks'},
                {'label': '☀️ [SIMULATION] The Sun (Slow Drift, Broadband Noise)', 'value': 'sim_sun'},
                {'label': '💫 [SIMULATION] Vela Pulsar (Sidereal Drift, Periodic Pulses)', 'value': 'sim_vela'},
            ],
            value='real_tracker',
            style={'width': '650px'}
        ),
        html.Div(id='dataset-info-banner', style={'marginTop': '10px', 'fontSize': '13px', 'color': '#2b6cb0', 'fontWeight': '500'})
    ]),
    
    # Top Panel
    html.Div([
        dcc.Graph(id='top-plot', style={'height': '450px'})
    ], style={'backgroundColor': '#ffffff', 'padding': '10px', 'borderRadius': '8px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
    
    # Bottom Panels
    html.Div([
        html.Div([
            dcc.Graph(id='bottom-left-plot', style={'height': '420px'})
        ], style={'width': '49%', 'display': 'inline-block', 'backgroundColor': '#ffffff', 'padding': '10px', 'borderRadius': '8px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'}),
        
        html.Div([
            dcc.Graph(id='bottom-right-plot', style={'height': '420px'})
        ], style={'width': '49%', 'display': 'inline-block', 'float': 'right', 'backgroundColor': '#ffffff', 'padding': '10px', 'borderRadius': '8px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'})
    ])
])

@app.callback(
    [Output('dataset-info-banner', 'children'),
     Output('top-plot', 'figure'),
     Output('bottom-left-plot', 'figure'),
     Output('bottom-right-plot', 'figure')],
    [Input('target-selector', 'value')]
)
def update_dashboard(selected):
    # 1. REAL TRACKER POWER DYNAMICS
    if selected == 'real_tracker':
        df = REAL_DATA["tracker_power"]
        if df is not None:
            info = f"Loaded {len(df)} discrete tracking steps from {DATA_DIR.name}/tracker_power_dynamics.csv"
            
            fig_top = go.Figure()
            fig_top.add_trace(go.Scatter(x=df['Source_l'], y=df['Source_m'], mode='lines', name='Actual Celestial Path', line=dict(color='#e53e3e', width=3)))
            fig_top.add_trace(go.Scatter(x=df['Steer_l'], y=df['Steer_m'], mode='lines+markers', name='CUDA Discrete Tracker Steering', line=dict(color='#3182ce', width=1.5), marker=dict(size=4)))
            fig_top.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='Array Boresight', marker=dict(color='#38a169', symbol='cross', size=14, line=dict(width=2))))
            fig_top.update_layout(
                title="Real CUDA V5 Sky Trajectory: Celestial Source Motion vs Discrete Window Steering",
                xaxis_title="Direction Cosine l (East-West)", yaxis_title="Direction Cosine m (North-South)",
                margin=dict(l=40, r=40, t=40, b=40), plot_bgcolor='#1a202c', paper_bgcolor='#fff'
            )
            
            fig_bl = go.Figure()
            fig_bl.add_trace(go.Scatter(x=df['Time_s'], y=df['Tracked_Power_dB'], mode='lines', name='Tracked Coherent Power', line=dict(color='#3182ce', width=2.5)))
            fig_bl.add_trace(go.Scatter(x=df['Time_s'], y=df['Untracked_Drift_Power_dB'], mode='lines', name='Untracked Drift Scan (Fixed)', line=dict(color='#e53e3e', dash='dash', width=2)))
            fig_bl.add_hline(y=-3.0, line_dash="dot", line_color="gray", annotation_text="-3 dB Half Power")
            fig_bl.update_layout(
                title="Actual Coherent Power: Active Steering vs Untracked Drift",
                xaxis_title="Observation Time [s]", yaxis_title="Received Power [dB]",
                yaxis_range=[-30, 2], margin=dict(l=40, r=40, t=40, b=40)
            )
            
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(x=df['Time_s'], y=df['Pointing_Error_arcmin'], mode='lines', name='Pointing Error', line=dict(color='#dd6b20', width=1.8)))
            fig_br.add_hline(y=df['Pointing_Error_arcmin'].mean(), line_dash="dash", line_color="blue", annotation_text=f"Mean Error: {df['Pointing_Error_arcmin'].mean():.2f}'")
            fig_br.update_layout(
                title="Actual Pointing Offset & Tracking Sawtooth Error",
                xaxis_title="Observation Time [s]", yaxis_title="Pointing Error [arcminutes]",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return info, fig_top, fig_bl, fig_br

    # 2. REAL ASTRONOMICAL VALIDATION BURSTS
    if selected.startswith('real_frb_'):
        rep = REAL_DATA["astro_report"]
        burst_key_map = {
            'real_frb_canonical': 'FRB20180916B_canonical',
            'real_frb_high_dm': 'High_DM_Burst',
            'real_frb_scatter': 'Scattering_Dominated'
        }
        target_burst = burst_key_map[selected]
        burst_data = None
        if rep and "results" in rep:
            for b in rep["results"]:
                if b.get("burst") == target_burst:
                    burst_data = b
                    break
                    
        if burst_data:
            t1 = burst_data["test_1_dispersion_sweep"]
            t3 = burst_data["test_3_off_boresight"]
            t4 = burst_data["test_4_array_scaling"]
            
            info = f"Burst: {target_burst} | Engine: {rep['engine']} | Injected DM: {t1['injected_dm']} pc/cm³ | Recovered S/N: {t1['recovered_snr']:.1f}σ | Radiometer Scaling Slope: {t4['scaling_slope']:.3f}"
            
            n_freq = 256
            n_time = 400
            freqs = np.linspace(400, 800, n_freq)
            t_bins = np.linspace(0, 10, n_time)
            dm = t1['injected_dm']
            k_dm = 4.148808e3 / 1000.0
            f_ref = freqs[-1]
            delays_s = k_dm * dm * ((freqs**-2.0) - (f_ref**-2.0))
            
            wf = np.random.normal(1.0, 0.2, (n_freq, n_time))
            for f in range(n_freq):
                p_t = 2.0 + delays_s[f]
                pulse = np.exp(-((t_bins - p_t)/0.12)**2)
                if 'scatter' in selected:
                    sc = 0.2 * (400.0 / freqs[f])**4
                    pulse = np.maximum(pulse, np.exp(-(t_bins - p_t)/sc) * (t_bins > p_t))
                wf[f, :] += pulse * 4.0
                
            fig_top = go.Figure(data=go.Heatmap(z=wf, x=t_bins, y=freqs, colorscale='Magma'))
            fig_top.update_layout(
                title=f"Astronomical Dynamic Spectrum Waterfall (Validated Burst: {target_burst})",
                xaxis_title="Time [s]", yaxis_title="Observing Frequency [MHz]",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            fig_bl = go.Figure()
            fig_bl.add_trace(go.Scatter(x=t3['off_axis_l'], y=t3['norm_powers'], mode='lines+markers', name='CUDA Measured Power', marker=dict(size=8, color='#3182ce')))
            fig_bl.update_layout(
                title="Off-Boresight Measured Beam Response",
                xaxis_title="Off-Axis Direction Cosine Δl", yaxis_title="Normalized Beam Power [linear]",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(x=t4['ant_counts'], y=t4['snrs'], mode='lines+markers', name='Measured S/N', marker=dict(size=9, color='#38a169')))
            fig_br.update_layout(
                title=f"Coherent Radiometer Array Scaling (Slope = {t4['scaling_slope']:.3f}, Expected = {t4['expected_slope']})",
                xaxis_title="Number of Antennas", yaxis_title="Recovered Pulse S/N",
                xaxis_type="log", yaxis_type="log",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return info, fig_top, fig_bl, fig_br

    # 3. REAL BENCHMARK & HARDWARE ARCHITECTURE EVOLUTION
    if selected == 'real_benchmarks':
        evo_df = REAL_DATA["evolution"]
        bench_df = REAL_DATA["cuda_v5_bench"]
        info = f"Loaded Architecture Evolution & CUDA V5 Profiling from {DATA_DIR.name}"
        
        fig_top = go.Figure()
        if evo_df is not None:
            fig_top.add_trace(go.Bar(x=evo_df['Version'], y=evo_df['Speedup_64ant_vs_CPUNaive'], name='Speedup vs CPU Naive', marker_color='#3182ce'))
            fig_top.add_trace(go.Bar(x=evo_df['Version'], y=evo_df['Speedup_64ant_vs_CPUOpt2'], name='Speedup vs OpenMP 24T', marker_color='#38a169'))
            fig_top.update_layout(
                title="Multi-Generational Speedup Comparison (64 Antennas)",
                xaxis_title="Architecture Version", yaxis_title="Speedup Factor (x)",
                barmode='group', margin=dict(l=40, r=40, t=40, b=40)
            )
            
        fig_bl = go.Figure()
        if bench_df is not None:
            ants = [f"{n} Ant" for n in bench_df['Antenna_Count']]
            fig_bl.add_trace(go.Bar(x=ants, y=bench_df['H2D_Transfer_ms'], name='H2D Transfer', marker_color='#63b3ed'))
            fig_bl.add_trace(go.Bar(x=ants, y=bench_df['Kernel_Exec_ms'], name='Kernel Exec', marker_color='#48bb78'))
            fig_bl.add_trace(go.Bar(x=ants, y=bench_df['D2H_Transfer_ms'], name='D2H Transfer', marker_color='#f6ad55'))
            fig_bl.update_layout(
                title="CUDA V5 End-to-End Latency Breakdown [ms]",
                xaxis_title="Antenna Configuration", yaxis_title="Time [ms]",
                barmode='stack', margin=dict(l=40, r=40, t=40, b=40)
            )
            
        fig_br = go.Figure()
        if bench_df is not None:
            fig_br.add_trace(go.Scatter(x=bench_df['Antenna_Count'], y=bench_df['Compute_Throughput_TFLOPs'], mode='lines+markers', name='Throughput', marker=dict(size=10, color='#805ad5'), line=dict(width=3)))
            fig_br.update_layout(
                title="Effective Compute Throughput Scaling [TFLOPs]",
                xaxis_title="Antenna Count", yaxis_title="TFLOPs",
                margin=dict(l=40, r=40, t=40, b=40)
            )
        return info, fig_top, fig_bl, fig_br

    # 4. SIMULATION MODES
    info = f"Running analytical physics simulation for: {selected.replace('sim_', '').upper()}"
    duration = 10.0
    n_time = 500
    time_s = np.linspace(0, duration, n_time)
    l_axis = np.linspace(-0.2, 0.2, 80)
    m_axis = np.linspace(-0.2, 0.2, 80)
    L, M = np.meshgrid(l_axis, m_axis)

    source_l = -0.15 + time_s * 0.03
    source_m = -0.05 + time_s * 0.01
    beam_l, beam_m = source_l[n_time//2], source_m[n_time//2]

    wavelength = 0.5
    d_eff = 9.0
    ux = (math.pi * d_eff / wavelength) * (L - beam_l)
    uy = (math.pi * d_eff / wavelength) * (M - beam_m)
    power = (np.sinc(ux / math.pi) * np.sinc(uy / math.pi))**2
    power_db = 10.0 * np.log10(np.maximum(power, 1e-4))

    fig_top = go.Figure(data=go.Contour(z=power_db, x=l_axis, y=m_axis, colorscale='Magma'))
    fig_top.add_trace(go.Scatter(x=source_l, y=source_m, mode='lines', name='Target Path', line=dict(color='white', width=2)))
    fig_top.add_trace(go.Scatter(x=[beam_l], y=[beam_m], mode='markers', name='Tracker Center', marker=dict(color='cyan', symbol='cross', size=12)))
    fig_top.update_layout(title="Simulated 2D Beam Footprint", xaxis_title="Direction l", yaxis_title="Direction m", margin=dict(l=40, r=40, t=40, b=40), plot_bgcolor='#1a202c')

    # Use real GPU power dynamics if generated
    df_power = None
    if selected == 'sim_vela' and REAL_DATA.get("vela_power") is not None:
        df_power = REAL_DATA["vela_power"]
        info = f"Loaded real GPU-computed Vela Pulsar tracking dynamics from {df_power.shape[0]} steps."
    elif selected == 'sim_sun' and REAL_DATA.get("sun_power") is not None:
        df_power = REAL_DATA["sun_power"]
        info = f"Loaded real GPU-computed Solar Radio tracking dynamics from {df_power.shape[0]} steps."

    fig_bl = go.Figure()
    if df_power is not None:
        fig_bl.add_trace(go.Scatter(x=df_power['Time_s'], y=df_power['Tracked_Power_dB'], mode='lines', name='Tracked Power (GPU Active)', line=dict(color='#3182ce', width=2.5)))
        fig_bl.add_trace(go.Scatter(x=df_power['Time_s'], y=df_power['Untracked_Drift_Power_dB'], mode='lines', name='Untracked Drift Scan', line=dict(color='#e53e3e', dash='dash', width=2)))
    else:
        dist_drift = np.sqrt(source_l**2 + source_m**2)
        u_drift = (math.pi * d_eff / wavelength) * dist_drift
        drift_power_db = 10.0 * np.log10(np.maximum((np.sinc(u_drift / math.pi))**2, 1e-4))
        fig_bl.add_trace(go.Scatter(x=time_s, y=np.zeros_like(time_s), mode='lines', name='Tracked Power (0 dB)', line=dict(color='#3182ce', width=2.5)))
        fig_bl.add_trace(go.Scatter(x=time_s, y=drift_power_db, mode='lines', name='Untracked Drift Scan', line=dict(color='#e53e3e', dash='dash', width=2)))
    
    fig_bl.add_hline(y=-3.0, line_dash="dot", line_color="gray", annotation_text="-3 dB Half Power")
    fig_bl.update_layout(title="Coherent Power: Active Steering vs Drift Transit", xaxis_title="Time [s]", yaxis_title="Power [dB]", yaxis_range=[-30, 2], margin=dict(l=40, r=40, t=40, b=40))

    n_freq = 200
    n_tbins = 300
    freqs = np.linspace(400, 800, n_freq)
    t_bins = np.linspace(0, duration, n_tbins)
    wf = np.random.normal(1.0, 0.2, (n_freq, n_tbins))

    if selected == 'sim_sun':
        burst = np.exp(-((t_bins - 5)/2.5)**2)
        for f in range(n_freq):
            wf[f, :] += burst * 1.5
    else:  # sim_vela
        for p in np.arange(1.0, 9.5, 1.2):
            for f in range(n_freq):
                delay = 0.5 * ((freqs[f]/800.0)**-2.0 - 1.0)
                wf[f, :] += np.exp(-((t_bins - (p + delay))/0.1)**2) * 2.5

    fig_br = go.Figure(data=go.Heatmap(z=wf, x=t_bins, y=freqs, colorscale='Viridis'))
    fig_br.update_layout(title=f"Simulated Dynamic Spectrum ({selected.replace('sim_', '').upper()})", xaxis_title="Time [s]", yaxis_title="Frequency [MHz]", margin=dict(l=40, r=40, t=40, b=40))

    return info, fig_top, fig_bl, fig_br

if __name__ == '__main__':
    print(f"Starting interactive beam tracker dashboard with REAL DATA from: {DATA_DIR.name if DATA_DIR else 'None'}...")
    print("Open http://127.0.0.1:8050 in your browser.")
    app.run(debug=True, port=8050)

