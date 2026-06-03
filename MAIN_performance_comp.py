"""
MAIN_performance.py - Blade aerodynamic assessment

Generates the standard set of plots and CSVs needed to evaluate a
wind turbine blade design:

  1. Geometry sanity check             -> blade_geometry.png
  2. Cp(lambda, beta) surface          -> cp_surface.png / .csv
  3. Cp(lambda) and Ct(lambda)         -> cp_ct_curves.png / cp_curve.csv
  4. Cp(lambda) family across pitch    -> cp_ct_pitch_family.png / .csv
  5. Power & thrust P(V), T(V)         -> power_thrust_curve.png / power_curve.csv
                                          (shows aerodynamic + electrical)
  6. Drivetrain efficiency             -> drivetrain_efficiency.png
  7. Loads at fixed RPM                -> loads_fixed_rpm.png / loads_curve.csv
  8. Section analysis at design point  -> sections_design_point.png / .csv
  9. Airfoil polars (raw + Viterna)    -> polar_<name>.png
 10. Validation vs reference BEM       -> validation_vs_reference.png /
                                          validation_table.csv
                                          (only runs if reference data is in
                                          the SETTINGS block below)

Edit only the SETTINGS block - everything below it is generic.
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from Geometria import BladeGeometry
from Aero      import Aero
from BEM       import BEMSolver
from controller import WindTurbineController


# ==================================================
# SETTINGS
# ==================================================

case_name  = "Smart_Blade"
output_dir = f"Outputs/{case_name}"
os.makedirs(output_dir, exist_ok=True)

# --- Geometry ---
# NOTE: switch back to the Smart_Blade geometry + polars that produced the
# earlier results you want to validate. Paths below are placeholders -
# replace with your actual Smart_Blade file paths.
geometry_file = "Inputs/smart/bladegeom_smart.csv"
n_sections    = 50
n_blades      = 3
hub_diameter  = 0.2

polar_files = {
    "cylinder": "Inputs/smart/cylinder.csv",
    "S823":     "Inputs/smart/S823.csv",
    "S822":     "Inputs/smart/S822.csv",
}

# --- Aerodynamic model ---
extrapolate_360             = True
apply_rotational_correction = False
blend_half_width_frac       = 0.05

# --- BEM solver ---
rho              = 1.225
max_iter         = 500
tol              = 1e-3
relaxation       = 0.15
use_tip_loss     = True
use_root_loss    = True
use_glauert      = True
use_root_cutoff  = False
root_cutoff_frac = 0.20

# --- Cp surface sweep ---
v_surface        = 8.0
lambda_values    = np.linspace(1.0, 14.0, 40)
pitch_values_deg = np.linspace(0.0, 30.0, 40)
cp_min_plot      = 0

user_lambda = None
user_pitch  = None

pitch_sweep_deg = [0.0, 3.0, 6.0, 10.0, 15.0, 20.0]

# --- Controller / power curve ---
v_cut_in          =  4.0
v_rated           = 12.0
v_cut_out         = 25.0
rpm_max           = 250.0
wind_speeds_power = np.arange(v_cut_in, v_cut_out + 1.0, 1.0)

weibull_c = 8.0
weibull_k = 2.0

generator_p_rated_W       = 5000.0
eta_drivetrain_at_rated   = 0.70
generator_const_loss_frac = 0.05

# --- Loads curve at fixed RPM, fixed pitch ---
# CHANGED: pitch was 5.0, now 0.0 to match the reference BEM analysis
# (which runs at fixed RPM=250 and pitch=0 deg). This makes loads_curve.csv
# directly comparable to the reference data.
pitch_loads_deg   = 0.0
wind_speeds_loads = np.arange(2.0, 21.0, 1.0)


# ==================================================
# VALIDATION DATA  (reference BEM at fixed RPM=250, pitch=0 deg)
# ==================================================
# Two reference curves: "rigid" (undeformed blade) and "deformed"
# (with aeroelastic deflection). Format: V[m/s], P[W], Cp, T[N], Q[N*m]
# Source: reference BEM analysis, same Smart_Blade geometry.
# Set REFERENCE_DATA = None to skip the validation block.

REFERENCE_DATA = {
    "rigid":    """\
2,-101.538211845175,-0.527683537236626,79.7033662491972,-4.06152847380701
3.28571428571429,41.0589096358336,0.048122928167582,188.455626434296,1.64235638543334
4.57142857142857,633.780816863056,0.275814628470321,339.172945438297,25.3512326745222
5.85714285714286,1950.35851747313,0.403544118884053,533.188034259282,78.0143406989254
7.14285714285714,3257.86271875822,0.371664352177787,662.698952861818,130.314508750329
8.42857142857143,4165.72521564801,0.289242912192146,732.092699727761,166.62900862592
9.71428571428572,4812.6096271735,0.218263964150382,771.458307903197,192.50438508694
11,4871.80236421697,0.152175811938369,777.679518805976,194.872094568679
12.2857142857143,5027.66458280057,0.112719507845605,813.684905140097,201.106583312023
13.5714285714286,5052.68067817761,0.0840386713857999,854.481271475291,202.107227127104
14.8571428571429,5279.86354668254,0.0669346230784778,921.518410536332,211.194541867302
16.1428571428571,5528.35396559178,0.0546372274156721,998.601179976962,221.134158623671
17.4285714285714,5622.42132445913,0.0441542016047557,1075.25959287283,224.896852978365
18.7142857142857,5802.1060408883,0.0368044147522785,1165.61829220859,232.084241635532
20,6035.55372000264,0.0313661455946154,1266.05350109845,241.422148800105""",

    "deformed": """\
2,-101.404407977224,-0.526988172436978,78.4872292643412,-4.05617631908897
3.28571428571429,31.2514179860893,0.0366280974389817,180.146566050658,1.25005671944357
4.57142857142857,594.248895477488,0.258610759373043,318.510651330296,23.7699558190995
5.85714285714286,1851.73239467723,0.383137618506784,495.0527089001,74.0692957870891
7.14285714285714,3153.80627292965,0.359793357336255,622.455191687189,126.152250917186
8.42857142857143,4170.33058520688,0.289562681364132,698.31080526903,166.813223408275
9.71428571428572,5098.45601684357,0.231227817647909,756.249612723477,203.938240673743
11,5649.14285484827,0.17645685034486,791.531090915792,225.965714193931
12.2857142857143,6089.66842037145,0.136529479240804,833.483343710013,243.586736814858
13.5714285714286,6314.9471438998,0.10503330838558,870.936003099276,252.597885755992
14.8571428571429,6579.99347031417,0.0834168116089015,926.454488203256,263.199738812567
16.1428571428571,6884.86565527876,0.0680437564011061,994.04342018867,275.39462621115
17.4285714285714,6995.69665480264,0.0549388568797066,1059.62765966113,279.827866192105
18.7142857142857,7254.3948940311,0.0460166974845319,1141.01198735055,290.175795761244
20,7632.32804204164,0.0396644158429327,1233.47024919578,305.293121681665""",
}


# ==================================================
# UTILITIES
# ==================================================

def save_fig(fig, fname):
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def n_nonconverged(result):
    return sum(1 for s in result["sections"] if not s["converged"])

def first_crossing(x, y, y_target):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    for i in range(len(y) - 1):
        if y[i] < y_target <= y[i + 1] and y[i + 1] != y[i]:
            t = (y_target - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None

def cp_lambda_sweep(lambda_array, beta_deg, v_inf, desc=""):
    rows = []
    for lam in tqdm(lambda_array, desc=f"Cp curve {desc}", leave=False):
        omega  = lam * v_inf / blade.R
        result = solver.solve_rotor(V_inf=v_inf, omega=omega,
                                    pitch=np.radians(beta_deg))
        rows.append({
            "lambda":      lam,
            "pitch_deg":   beta_deg,
            "v_inf":       v_inf,
            "rpm":         omega * 60.0 / (2.0 * np.pi),
            "Cp":          result["Cp"] if np.isfinite(result["Cp"]) else np.nan,
            "Ct":          result["Ct"] if np.isfinite(result["Ct"]) else np.nan,
            "power_W":     result["power"],
            "thrust_N":    result["thrust"],
            "non_converged": n_nonconverged(result),
        })
    return pd.DataFrame(rows)


# ==================================================
# 1. GEOMETRY
# ==================================================

blade = BladeGeometry(file_path=geometry_file, n_sections=n_sections,
                     B=n_blades, hub_diameter=hub_diameter)
blade.load_csv()
blade.Malla()
print(f"Blade: R = {blade.R:.4f} m, sections = {blade.n()}, B = {blade.B}")

_raw_geom = pd.read_csv(geometry_file)
_raw_geom.columns = _raw_geom.columns.str.strip()
_r_raw    = _raw_geom["r"].values + blade.r_hub
_c_raw    = _raw_geom["chord"].values
_t_raw    = _raw_geom["twist"].values
_af_raw   = _raw_geom["airfoil"].values

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.suptitle(f"Blade geometry sanity check - {case_name}\n"
             f"R = {blade.R:.3f} m,  hub = {hub_diameter/2:.3f} m,  "
             f"B = {blade.B},  sections = {blade.n()}",
             fontsize=12)

ax = axes[0]
ax.plot(_r_raw / blade.R, _c_raw, "o", ms=5, color="tomato", zorder=5,
        label="CSV stations")
ax.plot(blade.r / blade.R, blade.chord, "b-", lw=1.5,
        label=f"Resampled ({blade.n()} sections)")
ax.set_ylabel("Chord [m]")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(_r_raw / blade.R, _t_raw, "o", ms=5, color="tomato", zorder=5,
        label="CSV stations")
ax.plot(blade.r / blade.R, np.degrees(blade.twist), "g-", lw=1.5,
        label=f"Resampled ({blade.n()} sections)")
ax.set_ylabel("Twist [deg]")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
unique_airfoils = list(dict.fromkeys(_af_raw))
af_to_y = {name: i for i, name in enumerate(unique_airfoils)}
y_vals  = np.array([af_to_y[n] for n in _af_raw])
ax.step(_r_raw / blade.R, y_vals, where="post", color="purple", lw=1.5)
ax.scatter(_r_raw / blade.R, y_vals, color="purple", s=20, zorder=5)
ax.set_yticks(list(af_to_y.values()))
ax.set_yticklabels(list(af_to_y.keys()))
ax.set_ylabel("Airfoil")
ax.set_xlabel("r/R")
ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, "blade_geometry.png")


# ==================================================
# 2. AERO
# ==================================================

aero = Aero(geometry=blade, polar_files=polar_files,
            extrapolate_360=extrapolate_360, step_deg=1.0,
            blend_half_width_frac=blend_half_width_frac)
aero.load_polars()
if apply_rotational_correction:
    aero.correct_polars(
        tsr_ref=float(np.median(lambda_values)),
        v_ref  =v_surface,
    )

for name, polar in aero.polars.items():
    raw = aero.polars_raw[name]
    label_bem = "BEM polar (corrected)" if apply_rotational_correction else "BEM polar"
    a_min = raw["alpha"].min() - 2.0
    a_max = raw["alpha"].max() + 2.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Polar - {name}   [{case_name}]", fontsize=13)

    for col, (coef, ylabel) in enumerate(zip(["cl", "cd"], ["Cl", "Cd"])):
        ax = axes[0, col]
        ax.plot(polar["alpha"], polar[coef], lw=1.5, label=label_bem)
        ax.axvspan(a_min, a_max, color="orange", alpha=0.08,
                   label="Measured range")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("alpha [deg]"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} - full 360 deg")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(polar["alpha"], polar[coef], lw=1.5, label=label_bem)
        ax.plot(raw["alpha"], raw[coef], "o", ms=5, color="tomato",
                zorder=5, label="Raw data")
        ax.set_xlim(a_min, a_max)
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("alpha [deg]"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} - measured range")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, f"polar_{name}.png")


# ==================================================
# 3. SOLVER
# ==================================================

solver = BEMSolver(
    geometry=blade, aero=aero, rho=rho,
    max_iter=max_iter, tol=tol, relaxation=relaxation,
    use_tip_loss=use_tip_loss, use_root_loss=use_root_loss,
    use_glauert=use_glauert, use_root_cutoff=use_root_cutoff,
    root_cutoff_fraction=root_cutoff_frac,
)


# ==================================================
# 4. Cp(LAMBDA, BETA) SURFACE
# ==================================================

print(f"\nRunning Cp(lambda, beta) surface  "
      f"({len(lambda_values)} x {len(pitch_values_deg)} points)...")

cp_raw = np.full((len(lambda_values), len(pitch_values_deg)), np.nan)
for i, lam in enumerate(tqdm(lambda_values, desc="lambda sweep")):
    omega = lam * v_surface / blade.R
    for j, beta in enumerate(pitch_values_deg):
        result = solver.solve_rotor(V_inf=v_surface, omega=omega,
                                    pitch=np.radians(beta))
        if np.isfinite(result["Cp"]):
            cp_raw[i, j] = result["Cp"]

pd.DataFrame(cp_raw,
             index=pd.Index(lambda_values, name="lambda"),
             columns=pd.Index(pitch_values_deg, name="pitch_deg")
            ).to_csv(os.path.join(output_dir, "cp_surface.csv"))

i_opt, j_opt = np.unravel_index(np.nanargmax(cp_raw), cp_raw.shape)
lambda_opt   = lambda_values[i_opt]
pitch_opt    = pitch_values_deg[j_opt]
cp_opt       = cp_raw[i_opt, j_opt]

if user_lambda is not None and user_pitch is not None:
    i_usr  = np.argmin(np.abs(lambda_values    - user_lambda))
    j_usr  = np.argmin(np.abs(pitch_values_deg - user_pitch))
    lambda_active = user_lambda
    pitch_active  = user_pitch
    cp_active     = cp_raw[i_usr, j_usr]
    label_active  = "User point"
else:
    lambda_active = lambda_opt
    pitch_active  = pitch_opt
    cp_active     = cp_opt
    label_active  = "Optimum"

print(f"\nDesign point ({label_active}):  "
      f"lambda = {lambda_active:.3f},  beta = {pitch_active:.2f} deg,  "
      f"Cp = {cp_active:.4f}")


# ==================================================
# 5. Cp(LAMBDA) CURVES
# ==================================================

lambda_curve = np.linspace(lambda_values.min(), lambda_values.max(), 60)

print(f"\nCp curve at design pitch ({pitch_active:.2f} deg)...")
cp_curve_df = cp_lambda_sweep(lambda_curve, pitch_active, v_surface,
                              desc="design")
cp_curve_df.to_csv(os.path.join(output_dir, "cp_curve.csv"), index=False)

cp_pitch_sweep = {}
if pitch_sweep_deg:
    print(f"\nCp curve family ({len(pitch_sweep_deg)} pitch values)...")
    for beta in pitch_sweep_deg:
        cp_pitch_sweep[beta] = cp_lambda_sweep(
            lambda_curve, beta, v_surface, desc=f"beta={beta:.0f}")
    combined = pd.DataFrame({"lambda": lambda_curve})
    for beta, df in cp_pitch_sweep.items():
        combined[f"Cp_beta_{beta:g}"] = df["Cp"].values
        combined[f"Ct_beta_{beta:g}"] = df["Ct"].values
    combined.to_csv(os.path.join(output_dir, "cp_curve_pitch_sweep.csv"),
                    index=False)


# ==================================================
# 6. POWER CURVE (controller-driven)
# ==================================================

controller = WindTurbineController(
    solver=solver, lambda_design=lambda_active, pitch_design=pitch_active,
    rotor_radius=blade.R, v_rated=v_rated, v_cut_in=v_cut_in,
    v_cut_out=v_cut_out, rpm_max=rpm_max, pitch_control=True,
)
print(f"\n{controller}")

print("\nRunning power curve P(V)...")
power_df = controller.compute_power_curve(
    wind_speeds=wind_speeds_power,
    convergence_check_fn=n_nonconverged,
)

if generator_p_rated_W is not None:
    p_loss_const = generator_const_loss_frac * generator_p_rated_W
    p_aero_at_rated = generator_p_rated_W / eta_drivetrain_at_rated
    eta_max         = eta_drivetrain_at_rated * (
        1.0 + p_loss_const / p_aero_at_rated
    )
    p_aero  = power_df["power_W"].values
    eta_arr = np.where(p_aero > 0,
                       eta_max * p_aero / (p_aero + p_loss_const), 0.0)
    p_elec  = np.maximum(0.0, eta_arr * p_aero)
    p_elec  = np.minimum(p_elec, generator_p_rated_W)
    power_df["eta_drivetrain"] = eta_arr
    power_df["power_elec_W"]   = p_elec

power_df.to_csv(os.path.join(output_dir, "power_curve.csv"), index=False)

aep_kwh = controller.compute_aep(power_df, weibull_c, weibull_k)
if generator_p_rated_W is not None:
    v_arr   = power_df["v_inf"].values
    p_e_arr = np.maximum(power_df["power_elec_W"].values, 0.0)
    pdf     = (weibull_k / weibull_c) * (v_arr / weibull_c) ** (weibull_k - 1) \
              * np.exp(-(v_arr / weibull_c) ** weibull_k)
    aep_elec_kwh = 8760.0 * np.trapezoid(p_e_arr * pdf, v_arr) / 1e3
else:
    aep_elec_kwh = None


# ==================================================
# 7. LOADS CURVE  (fixed RPM, fixed pitch)
#    --- This is also the VALIDATION sweep ---
#    At RPM = rpm_max (250), pitch = pitch_loads_deg (now 0 deg)
#    Matches the reference BEM operating conditions exactly.
# ==================================================

print(f"\nLoads / validation sweep at RPM = {controller.rpm_rated:.1f}, "
      f"beta = {pitch_loads_deg} deg...")
loads_rows = []
for v in tqdm(wind_speeds_loads, desc="Loads", leave=False):
    result = solver.solve_rotor(V_inf=v, omega=controller.omega_rated,
                                pitch=np.radians(pitch_loads_deg))
    # Compute Cp here too, for direct comparison with reference
    A_disk = np.pi * blade.R**2
    p_avail = 0.5 * rho * A_disk * v**3
    Cp_val  = result["power"] / p_avail if p_avail > 0 else np.nan

    loads_rows.append({
        "v_inf":     v,
        "rpm":       controller.rpm_rated,
        "pitch_deg": pitch_loads_deg,
        "torque_Nm": result["torque"],
        "thrust_N":  result["thrust"],
        "power_W":   result["power"],
        "Cp":        Cp_val,
        "Ct":        result["Ct"],
        "non_converged": n_nonconverged(result),
    })
loads_df = pd.DataFrame(loads_rows)
loads_df.to_csv(os.path.join(output_dir, "loads_curve.csv"), index=False)


# ==================================================
# 8. SECTION ANALYSIS AT DESIGN POINT
# ==================================================

print("\nSection analysis at design point...")
omega_dp  = lambda_active * v_surface / blade.R
result_dp = solver.solve_rotor(V_inf=v_surface, omega=omega_dp,
                               pitch=np.radians(pitch_active))

sec_df = pd.DataFrame([{
    "r":         s["r"],
    "r_R":       s["r"] / blade.R,
    "alpha_deg": s["alpha_deg"],
    "Cl":        s["Cl"],
    "Cd":        s["Cd"],
    "a":         s["a"],
    "a_prime":   s["a_prime"],
    "F":         s["F"],
    "converged": s["converged"],
} for s in result_dp["sections"]])
sec_df.to_csv(os.path.join(output_dir, "sections_design_point.csv"),
              index=False)
n_nc = (~sec_df["converged"]).sum()
if n_nc > 0:
    print(f"  Note: {n_nc} non-converged sections")


# ==================================================
# PLOTS  (skipping repeated boilerplate for brevity in this comment;
#         everything between here and section 10 is identical to your
#         original file)
# ==================================================

# --- Cp surface ---
lambda_grid, beta_grid = np.meshgrid(lambda_values, pitch_values_deg,
                                     indexing="ij")
cp_plot = np.where(cp_raw < cp_min_plot, np.nan, cp_raw)
omega_max     = rpm_max * 2.0 * np.pi / 60.0
lambda_at_cap = omega_max * blade.R / v_surface

fig = plt.figure(figsize=(11, 7))
ax  = fig.add_subplot(111, projection="3d")
ax.set_box_aspect([1.5, 1.5, 0.8])
surf = ax.plot_surface(beta_grid, lambda_grid, cp_plot,
                       cmap="viridis", edgecolor="none", alpha=0.9)
if lambda_values.min() <= lambda_at_cap <= lambda_values.max():
    cp_at_cap = np.array([
        np.interp(lambda_at_cap, lambda_values, cp_plot[:, j])
        for j in range(len(pitch_values_deg))
    ])
    ax.plot(pitch_values_deg, np.full_like(pitch_values_deg, lambda_at_cap),
            cp_at_cap, color="black", lw=2.5, ls="--",
            label=(f"RPM cap = {rpm_max:.0f}: lambda <= {lambda_at_cap:.2f} "
                   f"(at V={v_surface:.0f} m/s)"))
ax.scatter([pitch_opt], [lambda_opt], [cp_opt], color="red", s=80, zorder=5,
           label=f"Optimum  Cp={cp_opt:.3f}")
if label_active == "User point":
    ax.scatter([pitch_active], [lambda_active], [cp_active], color="orange",
               s=100, marker="^", zorder=6,
               label=f"User point  Cp={cp_active:.3f}")
ax.set_xlabel("beta [deg]"); ax.set_ylabel("lambda"); ax.set_zlabel("Cp")
ax.set_title(f"Cp(lambda, beta) - {case_name}")
ax.view_init(elev=25, azim=-135)
ax.legend(loc="upper right", fontsize=9)
fig.colorbar(surf, ax=ax, shrink=0.5, label="Cp")
plt.tight_layout()
save_fig(fig, "cp_surface.png")

# --- Cp & Ct curves at design pitch ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Aerodynamic coefficients - {case_name}   "
             f"beta = {pitch_active:.2f} deg", fontsize=13)
ax = axes[0]
ax.plot(cp_curve_df["lambda"], cp_curve_df["Cp"], "b-o", ms=4, lw=1.5)
ax.axhline(16/27, color="k", lw=0.8, ls="--", label="Betz limit (0.593)")
ax.axvline(lambda_active, color="r", lw=0.8, ls="--",
           label=f"lambda = {lambda_active:.2f}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lambda"); ax.set_ylabel("Cp"); ax.set_title("Cp vs lambda")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax = axes[1]
ax.plot(cp_curve_df["lambda"], cp_curve_df["Ct"], "g-o", ms=4, lw=1.5)
ax.axvline(lambda_active, color="r", lw=0.8, ls="--",
           label=f"lambda = {lambda_active:.2f}")
ax.set_xlabel("lambda"); ax.set_ylabel("Ct"); ax.set_title("Ct vs lambda")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, "cp_ct_curves.png")

# --- Cp & Ct family across pitch ---
if cp_pitch_sweep:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Cp & Ct vs lambda - pitch family - {case_name}", fontsize=13)
    cmap = plt.cm.viridis
    n_curves = len(cp_pitch_sweep)
    colors = [cmap(i / max(n_curves - 1, 1)) for i in range(n_curves)]
    for ax, key, ylabel, title in [
        (axes[0], "Cp", "Cp", "Cp vs lambda"),
        (axes[1], "Ct", "Ct", "Ct vs lambda"),
    ]:
        for color, (beta, df) in zip(colors, cp_pitch_sweep.items()):
            ax.plot(df["lambda"], df[key], "-o", color=color, lw=1.5,
                    ms=3, label=f"beta = {beta:.1f} deg")
        if key == "Cp":
            ax.axhline(16/27, color="k", lw=0.8, ls="--",
                       label="Betz limit (0.593)")
        ax.set_xlabel("lambda"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "cp_ct_pitch_family.png")

# --- Power & thrust curve ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Power & thrust curve - {case_name}", fontsize=13)
ax = axes[0]
ax.plot(power_df["v_inf"], power_df["power_W"] / 1e3,
        "b-o", ms=5, lw=1.8, label="Aerodynamic (rotor shaft)", zorder=5)
ax.axhline(controller.p_rated / 1e3, color="b", lw=0.8, ls=":",
           label=f"P_rated aero = {controller.p_rated/1e3:.1f} kW")
if generator_p_rated_W is not None:
    ax.plot(power_df["v_inf"], power_df["power_elec_W"] / 1e3,
            "r-s", ms=5, lw=1.8, label="Electrical (after drivetrain)",
            zorder=4)
    ax.axhline(generator_p_rated_W / 1e3, color="r", lw=0.8, ls=":",
               label=f"P_rated elec = {generator_p_rated_W/1e3:.1f} kW (nameplate)")
    ax.fill_between(power_df["v_inf"], power_df["power_elec_W"] / 1e3,
                    power_df["power_W"] / 1e3, color="black",
                    alpha=0.10, hatch="//", lw=0, label="Drivetrain losses")
    v_at_nameplate = first_crossing(power_df["v_inf"].values,
                                    power_df["power_elec_W"].values,
                                    generator_p_rated_W)
    if v_at_nameplate is not None:
        ax.scatter([v_at_nameplate], [generator_p_rated_W / 1e3],
                   color="magenta", s=120, marker="*", zorder=10,
                   label=f"Nameplate reached @ V = {v_at_nameplate:.2f} m/s")
ax.axvline(v_rated, color="grey", lw=0.8, ls="--",
           label=f"V_rated = {v_rated} m/s")
ax.set_xlabel("Wind speed [m/s]"); ax.set_ylabel("Power [kW]")
ax.set_title("Power curve P(V)"); ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.3)
ax = axes[1]
ax.plot(power_df["v_inf"], power_df["thrust_N"], "g-o", ms=5, lw=1.8)
ax.axvline(v_rated, color="grey", lw=0.8, ls="--",
           label=f"V_rated = {v_rated} m/s")
ax.set_xlabel("Wind speed [m/s]"); ax.set_ylabel("Thrust [N]")
ax.set_title("Thrust curve T(V)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, "power_thrust_curve.png")

# --- Drivetrain efficiency ---
if generator_p_rated_W is not None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Drivetrain efficiency - {case_name}", fontsize=13)
    ax.plot(power_df["v_inf"], 100.0 * power_df["eta_drivetrain"],
            "k-o", ms=5, lw=1.5)
    ax.axhline(100.0 * eta_drivetrain_at_rated, color="r", lw=0.8, ls="--",
               label=f"Target at rated: {100*eta_drivetrain_at_rated:.0f}%")
    ax.axvline(v_rated, color="grey", lw=0.8, ls="--",
               label=f"V_rated = {v_rated} m/s")
    ax.set_xlabel("Wind speed [m/s]"); ax.set_ylabel("Drivetrain efficiency [%]")
    ax.set_ylim(0, 100); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "drivetrain_efficiency.png")

# --- Loads curve ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Loads at fixed RPM - {case_name}\n"
             f"RPM = {controller.rpm_rated:.1f},  "
             f"beta = {pitch_loads_deg} deg", fontsize=13)
axes[0].plot(loads_df["v_inf"], loads_df["torque_Nm"], "b-o", ms=5, lw=1.5)
axes[0].set_xlabel("Wind speed [m/s]"); axes[0].set_ylabel("Shaft torque [N*m]")
axes[0].set_title("Shaft torque"); axes[0].grid(True, alpha=0.3)
axes[1].plot(loads_df["v_inf"], loads_df["thrust_N"], "b-o", ms=5, lw=1.5)
axes[1].set_xlabel("Wind speed [m/s]"); axes[1].set_ylabel("Thrust [N]")
axes[1].set_title("Rotor thrust"); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, "loads_fixed_rpm.png")

# --- Sections plot ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    f"Radial section analysis - {case_name}   "
    f"lambda = {lambda_active:.2f},  beta = {pitch_active:.2f} deg,  "
    f"V = {v_surface} m/s", fontsize=12)
panels = [
    (axes[0, 0], "alpha_deg", "alpha [deg]",        "b"),
    (axes[0, 1], "Cl",        "Cl",                 "g"),
    (axes[0, 2], "Cd",        "Cd",                 "r"),
    (axes[1, 0], "a",         "Axial induction a",  "b"),
    (axes[1, 1], "a_prime",   "Tang. induction a'", "g"),
    (axes[1, 2], "F",         "Prandtl loss F",     "r"),
]
nc = sec_df[~sec_df["converged"]]
for ax, col, ylabel, color in panels:
    ax.plot(sec_df["r_R"], sec_df[col], f"{color}-o", ms=3, lw=1.5)
    ax.set_xlabel("r/R"); ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1); ax.grid(True, alpha=0.3)
    for x in nc["r_R"].values:
        ax.axvline(x, color="orange", lw=0.8, ls="--", alpha=0.5)
plt.tight_layout()
save_fig(fig, "sections_design_point.png")


# ==================================================
# 10. VALIDATION AGAINST REFERENCE BEM
# ==================================================
# Compares the loads_curve.csv we just generated (at RPM=250, pitch=0)
# against the reference BEM data in REFERENCE_DATA above.

if REFERENCE_DATA is not None:
    print("\n" + "=" * 60)
    print("  VALIDATION vs REFERENCE BEM")
    print("=" * 60)

    # Parse the embedded reference data
    ref = {}
    for label, txt in REFERENCE_DATA.items():
        ref[label] = pd.read_csv(
            io.StringIO(txt), header=None,
            names=["V", "P_W", "Cp", "T_N", "Q_Nm"],
        )

    # Build comparison table at the reference V values
    rows = []
    for _, ref_row in ref["rigid"].iterrows():
        v = ref_row.V
        # Skip negative reference points (V=2 m/s)
        if v < wind_speeds_loads.min() or v > wind_speeds_loads.max():
            continue
        # Interpolate your model onto the reference V
        my_P = float(np.interp(v, loads_df["v_inf"], loads_df["power_W"]))
        my_T = float(np.interp(v, loads_df["v_inf"], loads_df["thrust_N"]))
        my_Q = float(np.interp(v, loads_df["v_inf"], loads_df["torque_Nm"]))
        my_Cp = float(np.interp(v, loads_df["v_inf"], loads_df["Cp"]))

        ref_d = ref["deformed"]
        ref_d_P = float(np.interp(v, ref_d.V, ref_d.P_W))
        ref_d_T = float(np.interp(v, ref_d.V, ref_d.T_N))
        ref_d_Q = float(np.interp(v, ref_d.V, ref_d.Q_Nm))
        ref_d_Cp = float(np.interp(v, ref_d.V, ref_d.Cp))

        # Error w.r.t. rigid reference
        def pct_err(mine, ref_):
            if abs(ref_) < 1e-6:
                return np.nan
            return 100.0 * (mine - ref_) / ref_

        rows.append({
            "V_mps":        v,
            "P_yours_W":    my_P,
            "P_ref_rig_W":  ref_row.P_W,
            "P_ref_def_W":  ref_d_P,
            "P_err_rig_pct": pct_err(my_P, ref_row.P_W),
            "Cp_yours":     my_Cp,
            "Cp_ref_rig":   ref_row.Cp,
            "Cp_ref_def":   ref_d_Cp,
            "Cp_err_rig_pct": pct_err(my_Cp, ref_row.Cp),
            "T_yours_N":    my_T,
            "T_ref_rig_N":  ref_row.T_N,
            "T_ref_def_N":  ref_d_T,
            "T_err_rig_pct": pct_err(my_T, ref_row.T_N),
            "Q_yours_Nm":   my_Q,
            "Q_ref_rig_Nm": ref_row.Q_Nm,
            "Q_ref_def_Nm": ref_d_Q,
            "Q_err_rig_pct": pct_err(my_Q, ref_row.Q_Nm),
        })

    val_df = pd.DataFrame(rows)
    val_df.to_csv(os.path.join(output_dir, "validation_table.csv"),
                  index=False, float_format="%.4f")

    # Print compact comparison
    print(f"\n{'V':>5} | {'P_yours':>9} {'P_rigid':>9} {'P_defor':>9} {'err%':>6} | "
          f"{'Cp_y':>6} {'Cp_r':>6} {'err%':>6}")
    print("-" * 80)
    for r in rows:
        print(f"{r['V_mps']:>5.1f} | "
              f"{r['P_yours_W']:>9.0f} {r['P_ref_rig_W']:>9.0f} "
              f"{r['P_ref_def_W']:>9.0f} {r['P_err_rig_pct']:>+6.1f} | "
              f"{r['Cp_yours']:>6.3f} {r['Cp_ref_rig']:>6.3f} "
              f"{r['Cp_err_rig_pct']:>+6.1f}")

    # Median errors (more robust than mean to outliers near cut-in)
    print(f"\nMedian absolute error vs RIGID reference:")
    for col, label in [("P_err_rig_pct", "Power"),
                       ("Cp_err_rig_pct", "Cp"),
                       ("T_err_rig_pct", "Thrust"),
                       ("Q_err_rig_pct", "Torque")]:
        med = np.nanmedian(np.abs(val_df[col]))
        print(f"  {label:>8}: {med:5.1f}%")

    # 4-panel comparison plot: P, Cp, T, Q vs V
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Validation vs reference BEM - {case_name}\n"
                 f"RPM = {controller.rpm_rated:.0f}, beta = {pitch_loads_deg} deg",
                 fontsize=13)

    # Power
    ax = axes[0, 0]
    ax.plot(loads_df["v_inf"], loads_df["power_W"] / 1e3,
            "b-o", lw=1.8, ms=5, label="Your BEM")
    ax.plot(ref["rigid"].V,    ref["rigid"].P_W / 1e3,
            "k-x", lw=1.5, ms=7, label="Reference (rigid)")
    ax.plot(ref["deformed"].V, ref["deformed"].P_W / 1e3,
            "r-^", lw=1.5, ms=6, label="Reference (deformed)")
    ax.set_xlabel("Wind speed [m/s]"); ax.set_ylabel("Power [kW]")
    ax.set_title("Power"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Cp
    ax = axes[0, 1]
    ax.plot(loads_df["v_inf"], loads_df["Cp"],
            "b-o", lw=1.8, ms=5, label="Your BEM")
    ax.plot(ref["rigid"].V,    ref["rigid"].Cp,
            "k-x", lw=1.5, ms=7, label="Reference (rigid)")
    ax.plot(ref["deformed"].V, ref["deformed"].Cp,
            "r-^", lw=1.5, ms=6, label="Reference (deformed)")
    ax.axhline(16/27, color="grey", lw=0.6, ls=":", label="Betz")
    ax.set_xlabel("Wind speed [m/s]"); ax.set_ylabel("Cp [-]")
    ax.set_title("Power coefficient"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Thrust
    ax = axes[1, 0]
    ax.plot(loads_df["v_inf"], loads_df["thrust_N"],
            "b-o", lw=1.8, ms=5, label="Your BEM")
    ax.plot(ref["rigid"].V,    ref["rigid"].T_N,
            "k-x", lw=1.5, ms=7, label="Reference (rigid)")
    ax.plot(ref["deformed"].V, ref["deformed"].T_N,
            "r-^", lw=1.5, ms=6, label="Reference (deformed)")
    ax.set_xlabel("Wind speed [m/s]"); ax.set_ylabel("Thrust [N]")
    ax.set_title("Rotor thrust"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Torque
    ax = axes[1, 1]
    ax.plot(loads_df["v_inf"], loads_df["torque_Nm"],
            "b-o", lw=1.8, ms=5, label="Your BEM")
    ax.plot(ref["rigid"].V,    ref["rigid"].Q_Nm,
            "k-x", lw=1.5, ms=7, label="Reference (rigid)")
    ax.plot(ref["deformed"].V, ref["deformed"].Q_Nm,
            "r-^", lw=1.5, ms=6, label="Reference (deformed)")
    ax.set_xlabel("Wind speed [m/s]"); ax.set_ylabel("Torque [N·m]")
    ax.set_title("Shaft torque"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "validation_vs_reference.png")

    print(f"\nValidation outputs:")
    print(f"  - {output_dir}/validation_table.csv")
    print(f"  - {output_dir}/validation_vs_reference.png")


# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 52)
print(f"  RESULTS SUMMARY - {case_name}")
print("=" * 52)
print(f"  R               = {blade.R:.4f} m   (B = {blade.B})")
print(f"  --- Aerodynamic optimum ---")
print(f"  lambda_opt      = {lambda_opt:.3f}")
print(f"  beta_opt        = {pitch_opt:.2f} deg")
print(f"  Cp_opt          = {cp_opt:.4f}")
if label_active == "User point":
    print(f"  --- User-defined active point ---")
    print(f"  lambda_user     = {lambda_active:.3f}")
    print(f"  beta_user       = {pitch_active:.2f} deg")
    print(f"  Cp_user         = {cp_active:.4f}  "
          f"(delta = {cp_active - cp_opt:+.4f})")
print(f"  --- Controller ---")
print(f"  RPM_max         = {controller.rpm_rated:.1f}")
print(f"  V_transition    = {controller.v_transition:.2f} m/s")
print(f"  P_rated (aero)  = {controller.p_rated:.1f} W")
print(f"  Max thrust      = {power_df['thrust_N'].max():.1f} N")
print(f"  AEP (aero)      = {aep_kwh:.1f} kWh/year  "
      f"(Weibull c={weibull_c}, k={weibull_k})")
if generator_p_rated_W is not None:
    print(f"  --- Drivetrain & electrical ---")
    print(f"  Generator       = {generator_p_rated_W:.0f} W  "
          f"(eta_rated = {eta_drivetrain_at_rated:.2f})")
    p_elec_max = power_df["power_elec_W"].max()
    print(f"  Peak electrical = {p_elec_max:.1f} W")
    print(f"  AEP (elec)      = {aep_elec_kwh:.1f} kWh/year")
    print(f"  AEP loss to drive train = "
          f"{aep_kwh - aep_elec_kwh:.1f} kWh/year "
          f"({100*(aep_kwh - aep_elec_kwh)/aep_kwh:.1f}%)")
    v_at_nameplate = first_crossing(
        power_df["v_inf"].values, power_df["power_elec_W"].values,
        generator_p_rated_W)
    if v_at_nameplate is not None:
        print(f"  Nameplate ({generator_p_rated_W:.0f} W) "
              f"reached at V = {v_at_nameplate:.2f} m/s")
    else:
        print(f"  Nameplate ({generator_p_rated_W:.0f} W) never reached "
              f"(peak electrical = {p_elec_max:.1f} W)")
print("=" * 52)
print(f"\nOutputs in: {output_dir}")
