"""
MAIN_modelica.py
================

Aerodynamic-only driver for time-domain analyses that will be compared
with (or coupled to) a Modelica model.

Generates rotor power/torque/thrust traces for a prescribed input
trajectory (t, V_inf(t), omega(t), pitch(t)) using either:

    aero_mode = "steady"   →  quasi-steady BEM, evaluated at every time
                              step (no wake memory)
    aero_mode = "dynamic"  →  unsteady BEM with Øye 2-pole dynamic
                              inflow filter on axial induction
    aero_mode = "compare"  →  run both and produce side-by-side plots
                              (recommended for validation)

The script does NOT model:
    - generator electromagnetics
    - drivetrain torsion
    - tower fore/aft motion
    - controller dynamics

Those belong in Modelica. This script gives you the aerodynamic side
in isolation, so you can verify the Python aero against the Modelica
aero before closing the loop.

Inputs to the simulation are defined in the TRAJECTORY block below.
You can:
    - load CSV with columns t, V, omega_rad_s OR rpm, pitch_deg
    - or generate analytical traces (step in V, gust, pitch ramp, etc.)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Geometria      import BladeGeometry
from Aero           import Aero
from BEM            import BEMSolver
from unsteady_bem   import UnsteadyBEM


# ==================================================
# CASE SETTINGS
# ==================================================

case_name  = "Smart_Blade_Modelica"
output_dir = f"Outputs/{case_name}"
os.makedirs(output_dir, exist_ok=True)

# Geometry — same as MAIN_performance
geometry_file = "Inputs/bladegeom_smart.csv"
n_sections    = 50
n_blades      = 3
hub_diameter  = 0.2

polar_files = {
    "cylinder": "Inputs/cylinder.csv",
    "S822":     "Inputs/S822.csv",
    "S823":     "Inputs/S823.csv",
}

extrapolate_360             = True
apply_rotational_correction = True

rho                = 1.225
max_iter           = 200
tol                = 1e-3
relaxation         = 0.3
use_tip_loss       = True
use_root_loss      = True
use_glauert        = True
use_root_cutoff    = False
root_cutoff_frac   = 0.20

# For rotational correction the Aero needs a representative TSR
tsr_ref_for_correction = 7.0
v_ref_for_correction   = 8.0


# ==================================================
# AERO MODE: "steady" | "dynamic" | "compare"
# ==================================================

aero_mode = "compare"


# ==================================================
# TRAJECTORY — inputs to the rotor
# ==================================================
# Three options, pick one by setting trajectory_source:
#   "synthetic"    → generate analytically (step / gust / ramp)
#   "csv"          → load from CSV (columns: t, V, omega_rad_s, pitch_deg)
#   "modelica_io"  → load Modelica simulation outputs (same CSV format)

trajectory_source = "synthetic"

# --- synthetic trajectory parameters ---
t_end       = 60.0     # [s]
dt          = 0.05     # [s]  — 20 Hz
v_steady    = 8.0      # [m/s] base wind
omega_const = 23.0     # [rad/s] roughly 220 RPM, hold constant for now
pitch_const = 3.08     # [deg]  design pitch

# Wind profile: step at t_step from v_steady to v_step
t_step    = 20.0
v_step    = 12.0
t_step_back = 40.0     # second step back to v_steady

# --- CSV trajectory (used if trajectory_source == "csv") ---
trajectory_file = "Inputs/modelica_inputs.csv"


# ==================================================
# OUTPUT
# ==================================================

save_figures = True


# ==================================================
# BUILD GEOMETRY, AERO, SOLVER
# ==================================================

print("Building blade geometry & aero...")

blade = BladeGeometry(
    file_path=geometry_file, n_sections=n_sections,
    B=n_blades, hub_diameter=hub_diameter,
)
blade.load_csv()
blade.Malla()
print(f"Blade: R = {blade.R:.4f} m, sections = {blade.n()}, B = {blade.B}")

aero = Aero(geometry=blade, polar_files=polar_files,
            extrapolate_360=extrapolate_360, step_deg=1.0)
aero.load_polars()
if apply_rotational_correction:
    aero.correct_polars(
        tsr_ref=tsr_ref_for_correction,
        v_ref  =v_ref_for_correction,
    )

solver = BEMSolver(
    geometry=blade, aero=aero, rho=rho,
    max_iter=max_iter, tol=tol, relaxation=relaxation,
    use_tip_loss=use_tip_loss, use_root_loss=use_root_loss,
    use_glauert=use_glauert, use_root_cutoff=use_root_cutoff,
    root_cutoff_fraction=root_cutoff_frac,
)


# ==================================================
# BUILD TRAJECTORY
# ==================================================

if trajectory_source == "synthetic":
    t_arr = np.arange(0.0, t_end + dt, dt)
    n     = len(t_arr)

    # Wind: hold at v_steady, step up at t_step, step back at t_step_back
    V_arr = np.full(n, v_steady)
    V_arr[(t_arr >= t_step) & (t_arr < t_step_back)] = v_step

    omega_arr     = np.full(n, omega_const)
    pitch_deg_arr = np.full(n, pitch_const)

elif trajectory_source == "csv":
    df_in = pd.read_csv(trajectory_file)
    t_arr     = df_in["t"].values
    V_arr     = df_in["V"].values
    if "omega_rad_s" in df_in.columns:
        omega_arr = df_in["omega_rad_s"].values
    elif "rpm" in df_in.columns:
        omega_arr = df_in["rpm"].values * 2.0 * np.pi / 60.0
    else:
        raise KeyError("CSV must contain 'omega_rad_s' or 'rpm'")
    pitch_deg_arr = df_in["pitch_deg"].values

else:
    raise ValueError(f"Unknown trajectory_source: {trajectory_source}")

print(f"\nTrajectory: {len(t_arr)} samples, dt = {np.mean(np.diff(t_arr)):.3f} s, "
      f"t_end = {t_arr[-1]:.2f} s")
print(f"  V range:     {V_arr.min():.2f} to {V_arr.max():.2f} m/s")
print(f"  omega range: {omega_arr.min():.2f} to {omega_arr.max():.2f} rad/s")
print(f"  pitch range: {pitch_deg_arr.min():.2f} to {pitch_deg_arr.max():.2f} deg")


# ==================================================
# RUN SIMULATIONS
# ==================================================

results = {}

modes_to_run = ["steady", "dynamic"] if aero_mode == "compare" else [aero_mode]

for mode in modes_to_run:
    print(f"\n--- Running aero_mode = '{mode}' ---")
    use_dynamic = (mode == "dynamic")
    engine      = UnsteadyBEM(solver, dynamic=use_dynamic)

    df_out = engine.simulate(
        t_array         = t_arr,
        V_array         = V_arr,
        omega_array     = omega_arr,
        pitch_deg_array = pitch_deg_arr,
        verbose         = True,
    )
    df_out.to_csv(os.path.join(output_dir, f"timeseries_{mode}.csv"),
                  index=False)
    results[mode] = df_out
    print(f"  Saved timeseries_{mode}.csv  ({len(df_out)} rows)")


# ==================================================
# PLOTS
# ==================================================

def save_fig(fig, fname):
    if save_figures:
        fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Inputs panel
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
fig.suptitle(f"Trajectory inputs — {case_name}", fontsize=13)
axes[0].plot(t_arr, V_arr, "b-", lw=1.5)
axes[0].set_ylabel("V_inf [m/s]")
axes[0].grid(True, alpha=0.3)
axes[1].plot(t_arr, omega_arr * 60.0 / (2.0 * np.pi), "g-", lw=1.5)
axes[1].set_ylabel("RPM")
axes[1].grid(True, alpha=0.3)
axes[2].plot(t_arr, pitch_deg_arr, "r-", lw=1.5)
axes[2].set_ylabel("pitch [deg]")
axes[2].set_xlabel("Time [s]")
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, "trajectory_inputs.png")


# Outputs panel — one plot per quantity, overlay both modes if available
quantities = [
    ("power",  "Rotor power [kW]",   1e-3),
    ("torque", "Rotor torque [N·m]", 1.0),
    ("thrust", "Rotor thrust [N]",   1.0),
    ("Cp",     "Cp",                 1.0),
    ("Ct",     "Ct",                 1.0),
]

fig, axes = plt.subplots(len(quantities), 1, figsize=(11, 11), sharex=True)
fig.suptitle(f"Aerodynamic response — {case_name}", fontsize=13)

colors = {"steady": "tab:blue", "dynamic": "tab:red"}
labels = {"steady": "Quasi-steady BEM", "dynamic": "Dynamic inflow (Øye)"}

for ax, (col, ylabel, scale) in zip(axes, quantities):
    for mode, df in results.items():
        ax.plot(df["t"], df[col] * scale,
                lw=1.5, color=colors[mode], label=labels[mode])
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

axes[-1].set_xlabel("Time [s]")
plt.tight_layout()
save_fig(fig, "aero_response.png")


# Difference plot (only when comparing both modes)
if "steady" in results and "dynamic" in results:
    df_s = results["steady"]
    df_d = results["dynamic"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle(f"Dynamic - Steady deviation — {case_name}", fontsize=13)

    axes[0].plot(df_s["t"], (df_d["power"] - df_s["power"]) / 1e3,
                 "k-", lw=1.5)
    axes[0].axhline(0, color="grey", lw=0.5, ls="--")
    axes[0].set_ylabel("ΔP [kW]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_s["t"], df_d["thrust"] - df_s["thrust"], "k-", lw=1.5)
    axes[1].axhline(0, color="grey", lw=0.5, ls="--")
    axes[1].set_ylabel("ΔT [N]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "aero_response_difference.png")


# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 56)
print(f"  AERODYNAMIC SIMULATION SUMMARY — {case_name}")
print("=" * 56)
print(f"  Trajectory:   {len(t_arr)} samples, t = 0 to {t_arr[-1]:.1f} s")
print(f"  Modes run:    {list(results.keys())}")
for mode, df in results.items():
    print(f"  --- {mode} ---")
    print(f"    Mean power:    {df['power'].mean()/1e3:.3f} kW")
    print(f"    Peak power:    {df['power'].max()/1e3:.3f} kW")
    print(f"    Mean thrust:   {df['thrust'].mean():.1f} N")
    print(f"    Peak thrust:   {df['thrust'].max():.1f} N")
print("=" * 56)
print("\nSaved files:")
for mode in results:
    print(f"  {os.path.join(output_dir, f'timeseries_{mode}.csv')}")
print(f"  {os.path.join(output_dir, 'aero_response.png')}")
if "steady" in results and "dynamic" in results:
    print(f"  {os.path.join(output_dir, 'aero_response_difference.png')}")
