"""
MAIN_performance.py  — Blade aerodynamic assessment

Outputs:
  blade_geometry.png              polar_<name>.png
  cp_surface.png / cp_surface.csv
  cp_ct_curves.png / cp_curve.csv
  cp_ct_pitch_family.png / cp_curve_pitch_sweep.csv
  power_thrust_curve.png / power_curve_raw.csv / power_curve_filtered.csv
  drivetrain_efficiency.png
  loads_fixed_rpm.png / loads_curve.csv
  sections_design_point.png / sections_design_point.csv

Edit only the SETTINGS block.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from tqdm import tqdm
from scipy.interpolate import griddata

from Geometria import BladeGeometry
from Aero      import Aero
from BEM       import BEMSolver
from controller import WindTurbineController


# ==================================================
# SETTINGS
# ==================================================

case_name  = "Test3_ROT"
output_dir = f"Outputs/{case_name}"

# --- Geometry ---
geometry_file = "Inputs/smart/bladegeom_smart.csv"
n_sections    = 50
n_blades      = 3
hub_diameter  = 0.2
geometry_r_is_span_offset = True

polar_files = {
    "cylinder": "Inputs/smart/cylinder.csv",
    "S822":      "Inputs/smart/S822.csv",
    "S823":      "Inputs/smart/S823.csv"
}

# --- Aerodynamic model ---
extrapolate_360             = True
apply_rotational_correction = True
min_r_over_R_for_rot_corr = 0.15
max_r_over_R_for_rot_corr = 0.45
alpha_min_rot_corr_deg = -5
alpha_max_rot_corr_deg = 20
delta_cl_max = 0.2
apply_rot_corr_before_viterna = True
# Airfoil-transition blending width (±fraction of R).
# Increase to 0.10 if section plots look jagged at airfoil boundaries.
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
high_induction_model = "buhl"
# options: "buhl", "reference_glauert"
buhl_transition_model = "hard"
# options: "hard", "latched", "smooth_blend"
# Use buhl_transition_model="latched" with count=5 only as an optional
# high-loading stabilization sensitivity case, not as the baseline default.
buhl_deactivation_count_required = 5

# --- Official Cp-map validity policy ---
# Raw and filtered outputs are preserved. The official performance map and
# optimum use cp_surface_validated.csv, which applies stricter post-processing
# validity filters to remove off-design or physically suspicious steady-BEM
# points such as near-zero-pitch high-TSR negative-torque states.
max_a_valid = 0.6
max_K_valid = 1.0
max_CT_local_valid = 2.0

# --- Optional iteration-history diagnostics ---
# Disabled by default. When enabled, histories are saved only for the
# selected lambda-pitch cases and radial band below.
save_iteration_history = True
iteration_history_cases = [
    {"lambda": 8.2564, "pitch_deg": 3.0769, "label": "raw_optimum_failed"},
    {"lambda": 7.8974, "pitch_deg": 3.0769, "label": "failed_near_peak"},
    {"lambda": 7.5385, "pitch_deg": 3.0769, "label": "failed_lower_lambda"},
    {"lambda": 8.6154, "pitch_deg": 3.0769, "label": "valid_filtered_optimum"},
]
iteration_history_rR_min = 0.25
iteration_history_rR_max = 0.45

# --- Cp surface sweep ---
v_surface        = 8
lambda_values    = np.linspace(0.0, 14.0, 40)
pitch_values_deg = np.linspace(-5, 20, 40)
cp_min_plot      = 0      # Cp values below this are hidden in the surface plot

# --- Optional refined Cp sweep around the current valid optimum ---
run_refined_sweep = False
lambda_refined = np.linspace(7.5, 9.5, 80)
pitch_refined_deg = np.linspace(2.5, 4.0, 50)

# Override the design point manually (set both to None to use the surface peak).
user_lambda = None    # e.g. 6.0
user_pitch  = None    # e.g. 0.0

# --- Cp(lambda) family across pitch (set to [] to skip) ---
pitch_sweep_deg = [-3,-2,1,0,1,2,3,4,5]

# --- Controller / power curve ---
v_cut_in          =  4.0
v_rated           = 12.0
v_cut_out         = 25.0
rpm_max           = 250.0
wind_speeds_power = np.arange(v_cut_in, v_cut_out + 1.0, 1.0)

# AEP estimate (Weibull wind-speed distribution)
weibull_c = 8.0
weibull_k = 2.0

# --- Drivetrain ---
# Set generator_p_rated_W = None to show aerodynamic power only (no electrical overlay).
generator_p_rated_W       = 5000.0   # nameplate electrical power [W]
eta_drivetrain_at_rated   = 0.70     # electrical / aerodynamic efficiency at rated load
generator_const_loss_frac = 0.05     # constant losses as fraction of generator_p_rated_W

# --- Loads curve (fixed RPM, fixed pitch — for structural sizing) ---
pitch_loads_deg   = 5.0
wind_speeds_loads = np.arange(4.0, 26.0, 1.0)


# ==================================================
# UTILITY FUNCTIONS
# ==================================================

def save_fig(fig, fname):
    """Save figure to output_dir, show it, then close."""
    fig.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def n_nonconverged(result):
    """Count sections that did not converge in a rotor result dict."""
    return sum(1 for s in result["sections"] if not s["converged"])


def export_geometry_diagnostics(blade, geometry_file, output_dir):
    """Export radius convention and mesh diagnostics for validation."""
    mesh_df = pd.DataFrame({
        "section_index": np.arange(blade.n()),
        "r": blade.r,
        "r_over_R": blade.r / blade.R,
        "chord": blade.chord,
        "twist_deg": np.degrees(blade.twist),
    })
    mesh_df.to_csv(
        os.path.join(output_dir, "geometry_mesh_diagnostics.csv"),
        index=False,
    )

    pd.DataFrame([{
        "geometry_file": geometry_file,
        "geometry_r_is_span_offset": geometry_r_is_span_offset,
        "hub_diameter": blade.hub_diameter,
        "hub_radius": blade.r_hub,
        "R": blade.R,
        "r_min": float(np.min(blade.r)),
        "r_max": float(np.max(blade.r)),
        "r_min_over_R": float(np.min(blade.r) / blade.R),
        "n_sections": blade.n(),
        "B": blade.B,
    }]).to_csv(
        os.path.join(output_dir, "geometry_summary.csv"),
        index=False,
    )


def section_diagnostic_summary(result, n_sections):
    """Summarize section convergence diagnostics for one rotor result."""
    sections = result["sections"]
    n_nc = n_nonconverged(result)
    invalid_reason = "non_converged_sections" if n_nc > 0 else ""
    return {
        "n_nonconverged": n_nc,
        "non_converged": n_nc,
        "frac_nonconverged": n_nc / n_sections,
        "valid": n_nc == 0,
        "invalid_reason": invalid_reason,
        "max_iterations": max(s.get("iterations", np.nan) for s in sections),
        "max_a": max(s.get("a", np.nan) for s in sections),
        "max_K": max(s.get("K", np.nan) for s in sections),
        "max_abs_a_prime": max(abs(s.get("a_prime", np.nan)) for s in sections),
        "min_raw_discriminant": min(
            s.get("raw_discriminant", s.get("discriminant", np.nan))
            for s in sections
        ),
    }


def first_crossing(x, y, y_target):
    """
    Return the x-value where y first reaches y_target (linear interpolation).
    Returns None if y never reaches y_target.
    Handles flat segments (e.g. power clamped at nameplate) correctly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for i in range(len(y) - 1):
        if y[i] < y_target <= y[i + 1] and y[i + 1] != y[i]:
            t = (y_target - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


# ==================================================
# COMPUTATION FUNCTIONS
# ==================================================

def run_cp_surface(solver, blade, lambda_values, pitch_values_deg, v_surface):
    """
    Sweep Cp over the full (lambda, beta) grid.
    Returns cp_raw[i_lambda, j_pitch], surface_summary_df, section_diag_df.
    Non-finite Cp results are stored as NaN.
    """
    cp_raw = np.full((len(lambda_values), len(pitch_values_deg)), np.nan)
    summary_rows = []
    section_rows = []
    for i, lam in enumerate(tqdm(lambda_values, desc="lambda sweep")):
        omega = lam * v_surface / blade.R
        for j, beta in enumerate(pitch_values_deg):
            res = solver.solve_rotor(
                V_inf=v_surface, omega=omega, pitch=np.radians(beta)
            )
            if np.isfinite(res["Cp"]):
                cp_raw[i, j] = res["Cp"]

            sections = res["sections"]
            n_nonconv = n_nonconverged(res)
            n_invalid_discriminant = sum(
                1 for s in sections if s.get("discriminant_clipped", False)
            )
            n_physically_invalid = sum(
                1 for s in sections if not s.get("physically_valid", True)
            )
            summary_rows.append({
                "lambda": lam,
                "pitch_deg": beta,
                "Cp": res["Cp"] if np.isfinite(res["Cp"]) else np.nan,
                "Ct": res["Ct"] if np.isfinite(res["Ct"]) else np.nan,
                "torque": res["torque"],
                "thrust": res["thrust"],
                "n_nonconverged": n_nonconv,
                "n_invalid_discriminant": n_invalid_discriminant,
                "n_physically_invalid": n_physically_invalid,
                "max_K": max(s.get("K", np.nan) for s in sections),
                "max_a": max(s["a"] for s in sections),
                "min_a": min(s["a"] for s in sections),
                "max_abs_a_prime": max(abs(s["a_prime"]) for s in sections),
                "min_raw_discriminant": min(
                    s.get("raw_discriminant", np.nan) for s in sections
                ),
                "min_discriminant": min(
                    s.get("raw_discriminant", np.nan) for s in sections
                ),
                "max_iterations": max(s.get("iterations", np.nan) for s in sections),
                "n_glauert_active": sum(1 for s in sections if s.get("glauert_active", False)),
                "frac_physically_invalid": n_physically_invalid / len(sections),
            })

            for s in sections:
                section_rows.append({
                    "lambda": lam,
                    "pitch_deg": beta,
                    "r": s["r"],
                    "r_R": s["r"] / blade.R,
                    "alpha_deg": s["alpha_deg"],
                    "Cl": s["Cl"],
                    "Cd": s["Cd"],
                    "Cn": s["Cn"],
                    "Ct": s["Ct"],
                    "F": s["F"],
                    "sigma": s["sigma"],
                    "K": s.get("K", np.nan),
                    "CT_local": s.get("CT_local", np.nan),
                    "CT_local_over_F": s.get("CT_local_over_F", np.nan),
                    "fg": s.get("fg", np.nan),
                    "raw_discriminant": s.get("raw_discriminant", np.nan),
                    "clipped_discriminant": s.get("clipped_discriminant", np.nan),
                    "discriminant": s.get("raw_discriminant", np.nan),
                    "discriminant_clipped": s.get("discriminant_clipped", False),
                    "a": s["a"],
                    "a_prime": s["a_prime"],
                    "a_old": s.get("a_old", np.nan),
                    "a_raw": s.get("a_raw", np.nan),
                    "a_final": s.get("a_final", np.nan),
                    "high_induction_model": s.get("high_induction_model", ""),
                    "glauert_active": s.get("glauert_active", False),
                    "high_induction_active": s.get("high_induction_active", False),
                    "buhl_transition_model": s.get("buhl_transition_model", ""),
                    "buhl_blend_factor": s.get("buhl_blend_factor", np.nan),
                    "buhl_latched": s.get("buhl_latched", False),
                    "buhl_deactivation_counter": s.get(
                        "buhl_deactivation_counter", np.nan
                    ),
                    "a_momentum": s.get("a_momentum", np.nan),
                    "a_buhl": s.get("a_buhl", np.nan),
                    "iterations": s.get("iterations", np.nan),
                    "numerically_converged": s.get("numerically_converged", s["converged"]),
                    "physically_valid": s.get("physically_valid", True),
                    "invalid_reason": s.get("invalid_reason", ""),
                    "converged": s["converged"],
                })

    return cp_raw, pd.DataFrame(summary_rows), pd.DataFrame(section_rows)


def find_design_point(cp_raw, lambda_values, pitch_values_deg, user_lambda, user_pitch):
    """
    Locate the aerodynamic optimum and the active design point.

    Returns
    -------
    opt    : dict  {lambda, pitch, cp}        — surface peak (true maximum)
    active : dict  {lambda, pitch, cp, label} — user override or surface peak
    """
    if not np.isfinite(cp_raw).any():
        raise ValueError(
            "No finite Cp values are available for design-point selection. "
            "Relax the validation thresholds or inspect the filtered Cp surface."
        )
    i_opt, j_opt = np.unravel_index(np.nanargmax(cp_raw), cp_raw.shape)
    opt = {
        "lambda": float(lambda_values[i_opt]),
        "pitch":  float(pitch_values_deg[j_opt]),
        "cp":     float(cp_raw[i_opt, j_opt]),
    }
    if user_lambda is not None and user_pitch is not None:
        i_usr = int(np.argmin(np.abs(lambda_values    - user_lambda)))
        j_usr = int(np.argmin(np.abs(pitch_values_deg - user_pitch)))
        active = {
            "lambda": float(user_lambda),
            "pitch":  float(user_pitch),
            "cp":     float(cp_raw[i_usr, j_usr]),
            "label":  "User point",
        }
    else:
        active = {**opt, "label": "Optimum"}
    return opt, active


def build_filtered_cp_surface(summary_df, lambda_values, pitch_values_deg, high_induction_model):
    """
    Build a Cp surface with invalid lambda-beta cases masked as NaN.

    Buhl:
        valid if n_nonconverged == 0

    reference_glauert:
        valid if n_nonconverged == 0 and n_physically_invalid == 0
    """
    cp_filtered = np.full((len(lambda_values), len(pitch_values_deg)), np.nan)

    for _, row in summary_df.iterrows():
        lam = row["lambda"]
        beta = row["pitch_deg"]
        i = int(np.argmin(np.abs(lambda_values - lam)))
        j = int(np.argmin(np.abs(pitch_values_deg - beta)))

        valid = row["n_nonconverged"] == 0
        if high_induction_model == "reference_glauert":
            valid = valid and row.get("n_physically_invalid", 0) == 0

        if valid and np.isfinite(row["Cp"]):
            cp_filtered[i, j] = row["Cp"]

    return cp_filtered


def build_validated_cp_surface(summary_df, section_df, lambda_values,
                               pitch_values_deg, max_a_valid,
                               max_K_valid, max_CT_local_valid):
    """
    Build the official Cp surface for performance-map and optimum selection.

    This is a post-processing validity policy only. It does not modify the raw
    BEM solution, the raw Cp surface, or the basic convergence-filtered surface.
    """
    validated_df = summary_df.copy()

    max_ct_by_case = {}
    if section_df is not None and "CT_local" in section_df.columns:
        ct_df = section_df[np.isfinite(section_df["CT_local"])]
        if not ct_df.empty:
            max_ct_by_case = (
                ct_df.groupby(["lambda", "pitch_deg"])["CT_local"]
                .max()
                .to_dict()
            )

    cp_validated = np.full((len(lambda_values), len(pitch_values_deg)), np.nan)
    valid_flags = []
    invalid_reasons = []
    max_ct_values = []
    has_negative_cp = []
    has_negative_torque = []

    for _, row in validated_df.iterrows():
        lam = row["lambda"]
        beta = row["pitch_deg"]
        cp = row["Cp"]
        torque = row["torque"]
        max_a = row.get("max_a", np.nan)
        max_K = row.get("max_K", np.nan)
        n_nonconverged = row.get("n_nonconverged", 0)
        n_physically_invalid = row.get("n_physically_invalid", 0)
        max_ct = max_ct_by_case.get((lam, beta), np.nan)

        reasons = []
        if n_nonconverged > 0:
            reasons.append("non_converged_sections")
        if np.isfinite(cp) and cp < 0.0:
            reasons.append("negative_cp")
        if np.isfinite(torque) and torque < 0.0:
            reasons.append("negative_torque")
        if np.isfinite(max_a) and max_a > max_a_valid:
            reasons.append("max_a_above_threshold")
        if np.isfinite(max_K) and max_K > max_K_valid:
            reasons.append("max_K_above_threshold")
        if np.isfinite(max_ct) and max_ct > max_CT_local_valid:
            reasons.append("max_CT_local_above_threshold")
        if n_physically_invalid > 0:
            reasons.append("physically_invalid_sections")

        valid = len(reasons) == 0 and np.isfinite(cp)
        valid_flags.append(valid)
        invalid_reasons.append(";".join(reasons))
        max_ct_values.append(max_ct)
        has_negative_cp.append(bool(np.isfinite(cp) and cp < 0.0))
        has_negative_torque.append(bool(np.isfinite(torque) and torque < 0.0))

        if valid:
            i = int(np.argmin(np.abs(lambda_values - lam)))
            j = int(np.argmin(np.abs(pitch_values_deg - beta)))
            cp_validated[i, j] = cp

    validated_df["valid_for_performance_map"] = valid_flags
    validated_df["invalid_reason"] = invalid_reasons
    validated_df["max_CT_local"] = max_ct_values
    validated_df["has_negative_cp"] = has_negative_cp
    validated_df["has_negative_torque"] = has_negative_torque

    return cp_validated, validated_df


def save_validated_cp_surface(cp_validated, validated_df, lambda_array,
                              pitch_array, output_dir, suffix=""):
    """Save official post-processed Cp surface and validation diagnostics."""
    suffix_part = f"_{suffix}" if suffix else ""

    pd.DataFrame(
        cp_validated,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, f"cp_surface{suffix_part}_validated.csv"))

    validated_df.to_csv(
        os.path.join(output_dir, f"lambda_beta_summary{suffix_part}_validated.csv"),
        index=False,
    )


def build_startup_offdesign_cp_surface(summary_df, lambda_values, pitch_values_deg):
    """Build a diagnostic Cp surface that preserves off-design behavior."""
    cp_offdesign = np.full((len(lambda_values), len(pitch_values_deg)), np.nan)
    for _, row in summary_df.iterrows():
        if np.isfinite(row["Cp"]):
            i = int(np.argmin(np.abs(lambda_values - row["lambda"])))
            j = int(np.argmin(np.abs(pitch_values_deg - row["pitch_deg"])))
            cp_offdesign[i, j] = row["Cp"]
    return cp_offdesign


def save_startup_offdesign_cp_surface(cp_offdesign, lambda_array, pitch_array,
                                      output_dir, suffix=""):
    """Save startup / transition / off-design Cp surface for diagnostics."""
    suffix_part = f"_{suffix}" if suffix else ""
    pd.DataFrame(
        cp_offdesign,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(
        os.path.join(output_dir, f"cp_surface{suffix_part}_startup_offdesign.csv")
    )


def build_startup_offdesign_interpolated_cp_surface(cp_offdesign, lambda_array,
                                                    pitch_array):
    """
    Interpolate the diagnostic off-design surface while preserving negative Cp.

    This output is not used for official calculations.
    """
    lambda_grid, pitch_grid = np.meshgrid(lambda_array, pitch_array, indexing="ij")
    valid = np.isfinite(cp_offdesign)
    cp_interpolated = cp_offdesign.copy()
    mask = np.full(cp_offdesign.shape, 2, dtype=int)
    mask[valid] = 0
    if valid.sum() < 3:
        return cp_interpolated, mask

    source_points = np.column_stack((lambda_grid[valid], pitch_grid[valid]))
    source_values = cp_offdesign[valid]
    target_points = np.column_stack((lambda_grid.ravel(), pitch_grid.ravel()))
    interpolated_flat = griddata(
        source_points, source_values, target_points, method="linear"
    )
    interpolated = interpolated_flat.reshape(cp_offdesign.shape)
    fill = ~valid & np.isfinite(interpolated)
    cp_interpolated[fill] = interpolated[fill]
    mask[fill] = 1
    return cp_interpolated, mask


def save_startup_offdesign_interpolated_cp_surface(cp_interpolated, interpolation_mask,
                                                   lambda_array, pitch_array,
                                                   output_dir, suffix=""):
    """Save interpolated startup / off-design diagnostics."""
    suffix_part = f"_{suffix}" if suffix else ""
    pd.DataFrame(
        cp_interpolated,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(
        os.path.join(
            output_dir,
            f"cp_surface{suffix_part}_startup_offdesign_interpolated.csv",
        )
    )
    pd.DataFrame(
        interpolation_mask,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(
        os.path.join(
            output_dir,
            f"cp_surface{suffix_part}_startup_offdesign_interpolation_mask.csv",
        )
    )


def build_operating_classification_mask(validated_df, lambda_array, pitch_array):
    """Build an operating-envelope interpretation mask."""
    mask = np.full((len(lambda_array), len(pitch_array)), 1, dtype=int)
    for _, row in validated_df.iterrows():
        i = int(np.argmin(np.abs(lambda_array - row["lambda"])))
        j = int(np.argmin(np.abs(pitch_array - row["pitch_deg"])))
        reason = row["invalid_reason"]
        if bool(row["valid_for_performance_map"]):
            mask[i, j] = 0
        elif "non_converged_sections" in reason:
            mask[i, j] = 2
        elif bool(row["has_negative_cp"]) or bool(row["has_negative_torque"]):
            mask[i, j] = 3
        elif (
            "max_a_above_threshold" in reason
            or "max_K_above_threshold" in reason
            or "max_CT_local_above_threshold" in reason
        ):
            mask[i, j] = 4
    return mask


def save_operating_classification_mask(mask, lambda_array, pitch_array,
                                       output_dir, suffix=""):
    """Save operating-envelope interpretation mask."""
    suffix_part = f"_{suffix}" if suffix else ""
    pd.DataFrame(
        mask,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(
        os.path.join(output_dir, f"cp_surface{suffix_part}_operating_classification_mask.csv")
    )


def build_startup_offdesign_report(cp_pitch_sweep):
    """Summarize startup / transition / off-design pitch-sweep behavior."""
    rows = []
    for beta, df in cp_pitch_sweep.items():
        negative_cp = df[df["Cp"] < 0.0]
        negative_torque = df[df["torque_Nm"] < 0.0]
        non_converged = df[df["n_nonconverged"] > 0]
        invalid_reason_counts = (
            df.loc[df["invalid_reason"] != "", "invalid_reason"]
            .value_counts()
            .to_dict()
        )
        rows.append({
            "pitch_deg": beta,
            "first_lambda_negative_cp": (
                float(negative_cp["lambda"].min()) if not negative_cp.empty else np.nan
            ),
            "first_lambda_negative_torque": (
                float(negative_torque["lambda"].min())
                if not negative_torque.empty else np.nan
            ),
            "first_lambda_non_converged": (
                float(non_converged["lambda"].min())
                if not non_converged.empty else np.nan
            ),
            "max_a": df["max_a"].max(),
            "max_K": df["max_K"].max(),
            "max_CT_local": df["max_CT_local"].max(),
            "negative_cp_cases": int((df["Cp"] < 0.0).sum()),
            "negative_torque_cases": int((df["torque_Nm"] < 0.0).sum()),
            "non_converged_cases": int((df["n_nonconverged"] > 0).sum()),
            "invalid_reason_counts": ";".join(
                f"{k}:{v}" for k, v in sorted(invalid_reason_counts.items())
            ),
        })
    return pd.DataFrame(rows)


def optima_differ(opt_raw, opt_filtered, atol=1e-12):
    """True when raw and filtered optima differ in lambda, pitch, or Cp."""
    return (
        not np.isclose(opt_raw["lambda"], opt_filtered["lambda"], atol=atol)
        or not np.isclose(opt_raw["pitch"], opt_filtered["pitch"], atol=atol)
        or not np.isclose(opt_raw["cp"], opt_filtered["cp"], atol=atol)
    )


def save_cp_surface_outputs(cp_raw, cp_filtered, summary_df, section_df,
                            lambda_array, pitch_array, output_dir, suffix=""):
    """Save raw/filtered Cp surfaces and lambda-beta diagnostics."""
    suffix_part = f"_{suffix}" if suffix else ""

    pd.DataFrame(
        cp_raw,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, f"cp_surface{suffix_part}_raw.csv"))

    pd.DataFrame(
        cp_filtered,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, f"cp_surface{suffix_part}_filtered.csv"))

    summary_df.to_csv(
        os.path.join(output_dir, f"lambda_beta_summary{suffix_part}.csv"),
        index=False,
    )

    if section_df is not None:
        section_df.to_csv(
            os.path.join(output_dir, f"cp_surface{suffix_part}_sections_diagnostics.csv"),
            index=False,
        )


def build_interpolated_cp_surface(cp_filtered, lambda_array, pitch_array):
    """
    Interpolate missing non-negative Cp values for visualization only.

    Source data are only valid BEM points from cp_filtered with Cp >= 0.
    NaN values and negative Cp values are never used as interpolation
    sources. Negative Cp values are kept out of the visualization surface.

    Mask labels:
        0 = original valid non-negative BEM result
        1 = interpolated value
        2 = missing / outside interpolation region / excluded negative region
    """
    lambda_grid, pitch_grid = np.meshgrid(lambda_array, pitch_array, indexing="ij")
    finite = np.isfinite(cp_filtered)
    negative = finite & (cp_filtered < 0.0)
    valid = finite & (cp_filtered >= 0.0)

    cp_interpolated = cp_filtered.copy()
    cp_interpolated[negative] = np.nan
    mask = np.full(cp_filtered.shape, 2, dtype=int)
    mask[valid] = 0

    stats = {
        "valid_nonnegative_sources": int(valid.sum()),
        "excluded_negative_points": int(negative.sum()),
        "interpolated_points": 0,
        "remaining_nan": int(np.isnan(cp_interpolated).sum()),
        "negative_region_excluded": bool(
            not negative.any()
            or (
                np.isnan(cp_interpolated[negative]).all()
                and np.all(mask[negative] == 2)
            )
        ),
    }

    if valid.sum() < 3:
        return cp_interpolated, mask, stats

    source_points = np.column_stack((lambda_grid[valid], pitch_grid[valid]))
    source_values = cp_filtered[valid]
    target_points = np.column_stack((lambda_grid.ravel(), pitch_grid.ravel()))

    interpolated_flat = griddata(
        source_points, source_values, target_points, method="linear"
    )
    interpolated = interpolated_flat.reshape(cp_filtered.shape)
    interpolated[interpolated < 0.0] = np.nan

    fill = ~(valid | negative) & np.isfinite(interpolated)
    cp_interpolated[fill] = interpolated[fill]
    mask[fill] = 1
    cp_interpolated[cp_interpolated < 0.0] = np.nan

    stats["interpolated_points"] = int(fill.sum())
    stats["remaining_nan"] = int(np.isnan(cp_interpolated).sum())
    stats["negative_region_excluded"] = bool(
        not negative.any()
        or (
            np.isnan(cp_interpolated[negative]).all()
            and np.all(mask[negative] == 2)
        )
    )

    return cp_interpolated, mask, stats


def save_interpolated_cp_surface(cp_interpolated, interpolation_mask,
                                 lambda_array, pitch_array, output_dir,
                                 suffix=""):
    """Save traceable interpolated Cp surface and interpolation mask."""
    suffix_part = f"_{suffix}" if suffix else ""

    pd.DataFrame(
        cp_interpolated,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, f"cp_surface{suffix_part}_interpolated.csv"))

    pd.DataFrame(
        interpolation_mask,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, f"cp_surface{suffix_part}_interpolation_mask.csv"))


def build_interpolated_cp_visual_surface(cp_interpolated, interpolation_mask):
    """
    Build a zero-filled surface for plotting only.

    The traceable interpolation result is left unchanged. NaN and negative Cp
    values are replaced with zero only in this separate visualization surface.

    Visual mask labels:
        0 = original valid BEM point
        1 = interpolated point
        2 = invalid / missing / NaN region filled as zero for visualization
        3 = negative Cp region filled as zero for visualization
    """
    cp_visual = np.where(np.isnan(cp_interpolated), 0.0, cp_interpolated)
    cp_visual = np.where(cp_visual < 0.0, 0.0, cp_visual)

    visual_mask = interpolation_mask.copy()
    visual_mask[np.isnan(cp_interpolated)] = 2
    visual_mask[np.isfinite(cp_interpolated) & (cp_interpolated < 0.0)] = 3

    stats = {
        "zero_filled_nan_points": int(np.isnan(cp_interpolated).sum()),
        "zero_filled_negative_points": int(
            (np.isfinite(cp_interpolated) & (cp_interpolated < 0.0)).sum()
        ),
    }
    return cp_visual, visual_mask, stats


def save_interpolated_cp_visual_surface(cp_visual, visual_mask,
                                        lambda_array, pitch_array, output_dir,
                                        suffix=""):
    """Save zero-filled interpolation artifacts used for plotting only."""
    suffix_part = f"_{suffix}" if suffix else ""

    pd.DataFrame(
        cp_visual,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(
        os.path.join(output_dir, f"cp_surface{suffix_part}_interpolated_visual.csv")
    )

    pd.DataFrame(
        visual_mask,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(
        os.path.join(output_dir, f"cp_surface{suffix_part}_interpolation_mask.csv")
    )


def transition_plot_label(buhl_transition_model):
    """Human-readable transition model label for plots."""
    labels = {
        "hard": "Hard transition",
        "latched": "Latched transition",
        "smooth_blend": "Smooth blend transition",
    }
    return labels.get(buhl_transition_model, str(buhl_transition_model))


def save_cp_validity_mask_and_stats(summary_df, lambda_array, pitch_array,
                                    output_dir, suffix=""):
    """
    Save validity mask and summary stats for a Cp sweep.

    Mask labels:
        0 = valid
        1 = filtered / invalid
    """
    suffix_part = f"_{suffix}" if suffix else ""
    validity_mask = np.ones((len(lambda_array), len(pitch_array)), dtype=int)

    for _, row in summary_df.iterrows():
        i = int(np.argmin(np.abs(lambda_array - row["lambda"])))
        j = int(np.argmin(np.abs(pitch_array - row["pitch_deg"])))
        valid = (
            row["n_nonconverged"] == 0
            and row.get("n_physically_invalid", 0) == 0
        )
        validity_mask[i, j] = 0 if valid else 1

    pd.DataFrame(
        validity_mask,
        index=pd.Index(lambda_array, name="lambda"),
        columns=pd.Index(pitch_array, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, f"cp_surface{suffix_part}_validity_mask.csv"))

    total_cases = int(validity_mask.size)
    valid_cases = int((validity_mask == 0).sum())
    filtered_cases = total_cases - valid_cases
    stats = pd.DataFrame([{
        "total_cases": total_cases,
        "valid_cases": valid_cases,
        "filtered_cases": filtered_cases,
        "percentage_filtered": 100.0 * filtered_cases / total_cases,
    }])
    stats.to_csv(
        os.path.join(output_dir, f"cp_surface{suffix_part}_validity_summary.csv"),
        index=False,
    )


def run_cp_lambda_sweep(solver, blade, lambda_array, beta_deg, v_inf, desc=""):
    """
    Run Cp(lambda) at a fixed pitch angle.

    Returns a DataFrame with columns:
        lambda, pitch_deg, v_inf, rpm, Cp, Ct, power_W, thrust_N, non_converged
    """
    rows = []
    for lam in tqdm(lambda_array, desc=f"Cp curve {desc}", leave=False):
        omega = lam * v_inf / blade.R
        res   = solver.solve_rotor(V_inf=v_inf, omega=omega, pitch=np.radians(beta_deg))
        diag = section_diagnostic_summary(res, blade.n())
        section_rows = res.get("sections", [])
        ct_local_vals = [
            s.get("CT_local", np.nan) for s in section_rows
            if np.isfinite(s.get("CT_local", np.nan))
        ]
        rows.append({
            "lambda":        lam,
            "pitch_deg":     beta_deg,
            "v_inf":         v_inf,
            "rpm":           omega * 60.0 / (2.0 * np.pi),
            "Cp":            res["Cp"]  if np.isfinite(res["Cp"])  else np.nan,
            "Ct":            res["Ct"]  if np.isfinite(res["Ct"])  else np.nan,
            "power_W":       res["power"],
            "torque_Nm":     res["torque"],
            "thrust_N":      res["thrust"],
            "n_nonconverged": diag["n_nonconverged"],
            "non_converged":  diag["non_converged"],
            "valid":          diag["valid"],
            "invalid_reason": diag["invalid_reason"],
            "max_a":          diag["max_a"],
            "max_K":          diag["max_K"],
            "max_CT_local":   max(ct_local_vals) if ct_local_vals else np.nan,
        })
    return pd.DataFrame(rows)


def run_loads_curve(solver, wind_speeds, omega_rated, pitch_loads_deg):
    """
    Run a fixed-RPM, fixed-pitch loads sweep.

    Returns a DataFrame with columns:
        v_inf, rpm, pitch_deg, torque_Nm, thrust_N, power_W
    """
    rows = []
    for v in tqdm(wind_speeds, desc="Loads", leave=False):
        res = solver.solve_rotor(
            V_inf=v, omega=omega_rated, pitch=np.radians(pitch_loads_deg)
        )
        diag = section_diagnostic_summary(res, solver.geometry.n())
        rows.append({
            "v_inf":     v,
            "rpm":       omega_rated * 60.0 / (2.0 * np.pi),
            "pitch_deg": pitch_loads_deg,
            "torque_Nm": res["torque"],
            "thrust_N":  res["thrust"],
            "power_W":   res["power"],
            "non_converged": diag["non_converged"],
            "frac_nonconverged": diag["frac_nonconverged"],
            "valid": diag["valid"],
            "invalid_reason": diag["invalid_reason"],
            "max_iterations": diag["max_iterations"],
            "max_K": diag["max_K"],
            "min_raw_discriminant": diag["min_raw_discriminant"],
        })
    return pd.DataFrame(rows)


def run_section_analysis(solver, blade, lambda_active, pitch_active, v_surface):
    """
    Solve BEM at the design point and return per-section aerodynamic data.

    Returns a DataFrame with columns:
        r, r_R, alpha_deg, Cl, Cd, a, a_prime, F, converged
    """
    omega  = lambda_active * v_surface / blade.R
    result = solver.solve_rotor(
        V_inf=v_surface, omega=omega, pitch=np.radians(pitch_active)
    )
    return pd.DataFrame([{
        "r":         s["r"],
        "r_R":       s["r"] / blade.R,
        "alpha_deg": s["alpha_deg"],
        "Cl":        s["Cl"],
        "Cd":        s["Cd"],
        "a":         s["a"],
        "a_prime":   s["a_prime"],
        "F":         s["F"],
        "Cn":        s["Cn"],
        "Ct":        s["Ct"],
        "CT_local":  s.get("CT_local", np.nan),
        "CT_local_over_F": s.get("CT_local_over_F", np.nan),
        "sigma":     s["sigma"],
        "K":         s.get("K", np.nan),
        "fg":        s.get("fg", np.nan),
        "raw_discriminant": s.get("raw_discriminant", np.nan),
        "clipped_discriminant": s.get("clipped_discriminant", np.nan),
        "discriminant": s.get("raw_discriminant", np.nan),
        "discriminant_clipped": s.get("discriminant_clipped", False),
        "a_old":     s.get("a_old", np.nan),
        "a_raw":     s.get("a_raw", np.nan),
        "a_final":   s.get("a_final", np.nan),
        "high_induction_model": s.get("high_induction_model", ""),
        "glauert_active": s.get("glauert_active", False),
        "high_induction_active": s.get("high_induction_active", False),
        "buhl_transition_model": s.get("buhl_transition_model", ""),
        "buhl_blend_factor": s.get("buhl_blend_factor", np.nan),
        "buhl_latched": s.get("buhl_latched", False),
        "buhl_deactivation_counter": s.get(
            "buhl_deactivation_counter", np.nan
        ),
        "a_momentum": s.get("a_momentum", np.nan),
        "a_buhl": s.get("a_buhl", np.nan),
        "iterations": s.get("iterations", np.nan),
        "dFn":       s.get("dFn", np.nan),
        "dFt":       s.get("dFt", np.nan),
        "dQ":        (
            s.get("dFt", np.nan) * s["r"]
            if np.isfinite(s.get("dFt", np.nan)) else np.nan
        ),
        "numerically_converged": s.get("numerically_converged", s["converged"]),
        "physically_valid": s.get("physically_valid", True),
        "invalid_reason": s.get("invalid_reason", ""),
        "converged": s["converged"],
    } for s in result["sections"]])


def run_iteration_history_diagnostics(
    solver,
    blade,
    cases,
    v_surface,
    rR_min,
    rR_max,
):
    """
    Record per-iteration induction diagnostics for selected cases only.

    This is intentionally opt-in because it can create large files if used
    over a full Cp surface.
    """
    rows = []
    previous_record_state = getattr(solver, "record_iteration_history", False)
    solver.record_iteration_history = True

    try:
        for case in cases:
            lam = case["lambda"]
            pitch_deg = case["pitch_deg"]
            case_label = case.get("label", "")

            omega = lam * v_surface / blade.R
            result = solver.solve_rotor(
                V_inf=v_surface,
                omega=omega,
                pitch=np.radians(pitch_deg),
            )

            for section in result["sections"]:
                r_R = section["r"] / blade.R
                if r_R < rR_min or r_R > rR_max:
                    continue

                for item in section.get("iteration_history", []):
                    rows.append({
                        "case_label": case_label,
                        "lambda": lam,
                        "pitch_deg": pitch_deg,
                        "section_converged": section.get("converged", False),
                        "section_iterations": section.get("iterations", np.nan),
                        "start_label": item.get("start_label", ""),
                        "iteration": item.get("iteration", np.nan),
                        "r": item.get("r", np.nan),
                        "r_R": item.get("r_R", np.nan),
                        "a_old": item.get("a_old", np.nan),
                        "a_new": item.get("a_new", np.nan),
                        "a_relaxed": item.get("a_relaxed", np.nan),
                        "a_prime_old": item.get("a_prime_old", np.nan),
                        "a_prime_new": item.get("a_prime_new", np.nan),
                        "a_prime_relaxed": item.get("a_prime_relaxed", np.nan),
                        "phi_deg": item.get("phi_deg", np.nan),
                        "alpha_deg": item.get("alpha_deg", np.nan),
                        "Cl": item.get("Cl", np.nan),
                        "Cd": item.get("Cd", np.nan),
                        "Cn": item.get("Cn", np.nan),
                        "Ct": item.get("Ct", np.nan),
                        "K": item.get("K", np.nan),
                        "CT_local": item.get("CT_local", np.nan),
                        "CT_local_over_F": item.get("CT_local_over_F", np.nan),
                        "F": item.get("F", np.nan),
                        "residual_a": item.get("residual_a", np.nan),
                        "residual_a_prime": item.get("residual_a_prime", np.nan),
                        "high_induction_active": item.get(
                            "high_induction_active", False
                        ),
                        "buhl_transition_model": item.get(
                            "buhl_transition_model", ""
                        ),
                        "buhl_blend_factor": item.get("buhl_blend_factor", np.nan),
                        "buhl_latched": item.get("buhl_latched", False),
                        "buhl_deactivation_counter": item.get(
                            "buhl_deactivation_counter", np.nan
                        ),
                        "a_momentum": item.get("a_momentum", np.nan),
                        "a_buhl": item.get("a_buhl", np.nan),
                    })
    finally:
        solver.record_iteration_history = previous_record_state

    return pd.DataFrame(rows)


def apply_drivetrain(power_df, generator_p_rated_W,
                     eta_drivetrain_at_rated, generator_const_loss_frac):
    """
    Compute and attach drivetrain columns to power_df in-place.

    Model: load-dependent generator efficiency
        eta(P) = eta_max * P / (P + P_loss_const)

    Calibration: at P_aero = generator_p_rated_W / eta_at_rated,
    the output efficiency equals eta_drivetrain_at_rated.
    Excess electrical power is clamped at the nameplate rating.

    Adds columns: eta_drivetrain, power_elec_W
    """
    p_loss_const    = generator_const_loss_frac * generator_p_rated_W
    p_aero_at_rated = generator_p_rated_W / eta_drivetrain_at_rated
    eta_max         = eta_drivetrain_at_rated * (1.0 + p_loss_const / p_aero_at_rated)

    p_aero  = power_df["power_W"].values
    eta_arr = np.where(p_aero > 0, eta_max * p_aero / (p_aero + p_loss_const), 0.0)
    p_elec  = np.minimum(np.maximum(0.0, eta_arr * p_aero), generator_p_rated_W)

    power_df["eta_drivetrain"] = eta_arr
    power_df["power_elec_W"]   = p_elec


def compute_aep_electrical(power_df, weibull_c, weibull_k):
    """
    Compute electrical AEP (kWh/year) from power_elec_W using a Weibull PDF.
    Uses trapezoidal integration: AEP = 8760 * integral(P_elec * f(V) dV).
    """
    v   = power_df["v_inf"].values
    p   = np.maximum(power_df["power_elec_W"].values, 0.0)
    pdf = ((weibull_k / weibull_c) * (v / weibull_c) ** (weibull_k - 1)
           * np.exp(-(v / weibull_c) ** weibull_k))
    return 8760.0 * np.trapezoid(p * pdf, v) / 1e3


def add_power_curve_validity(power_df, n_sections):
    """Add validity labels to a raw power curve without changing controller output."""
    power_df = power_df.copy()
    if "non_converged" not in power_df.columns:
        power_df["non_converged"] = 0

    power_df["frac_nonconverged"] = power_df["non_converged"] / n_sections
    power_df["valid"] = power_df["non_converged"] == 0
    power_df["invalid_reason"] = np.where(
        power_df["valid"], "", "non_converged_sections"
    )
    return power_df


def build_filtered_power_curve(power_df):
    """
    Mask aerodynamic/electrical outputs where the operating point is invalid.
    Operating-point columns and validity diagnostics are retained.
    """
    filtered = power_df.copy()
    output_cols = [
        "Cp", "Ct", "power_W", "torque_Nm", "thrust_N",
        "eta_drivetrain", "power_elec_W",
    ]
    invalid = ~filtered["valid"]
    for col in output_cols:
        if col in filtered.columns:
            filtered.loc[invalid, col] = np.nan
    return filtered


def interpolate_power_curve_for_aep(filtered_power_df, power_col):
    """
    Fill invalid power values by interpolation over valid rows for AEP only.

    Returns
    -------
    aep_df : DataFrame
        Copy with power_col interpolated over v_inf.
    used_interpolation : bool
        True when one or more invalid/missing rows were filled.
    """
    aep_df = filtered_power_df.copy()
    missing = aep_df[power_col].isna()
    used_interpolation = bool(missing.any())
    interp_col = f"{power_col}_interpolated_for_aep"
    aep_df[interp_col] = False

    if used_interpolation:
        valid = aep_df[power_col].notna()
        if valid.sum() < 2:
            raise ValueError(
                f"Cannot compute filtered AEP for {power_col}: fewer than two valid points"
            )
        aep_df[power_col] = np.interp(
            aep_df["v_inf"].values,
            aep_df.loc[valid, "v_inf"].values,
            aep_df.loc[valid, power_col].values,
        )
        aep_df.loc[missing, interp_col] = True

    return aep_df, used_interpolation


def print_invalid_power_curve_rows(power_df):
    """Print invalid power-curve rows with enough context for diagnostics."""
    invalid = power_df[~power_df["valid"]]
    if invalid.empty:
        return

    print("\n  Warning: invalid power-curve operating points were found:")
    for _, row in invalid.iterrows():
        print(
            f"    V={row['v_inf']:.2f} m/s, "
            f"lambda={row['lambda']:.3f}, "
            f"pitch={row['pitch_deg']:.3f} deg, "
            f"Cp={row['Cp']:.4f}, "
            f"power={row['power_W']:.1f} W, "
            f"non_converged={int(row['non_converged'])}"
        )


# ==================================================
# PLOT FUNCTIONS
# ==================================================

def plot_blade_geometry(blade, geometry_file, case_name):
    """Plot chord, twist, and airfoil layout. Saves blade_geometry.png."""
    raw    = pd.read_csv(geometry_file)
    r_raw  = raw["r"].values + blade.r_hub
    c_raw  = raw["chord"].values
    t_raw  = raw["twist"].values
    af_raw = raw["airfoil"].values

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"Blade geometry sanity check - {case_name}\n"
        f"R = {blade.R:.3f} m,  hub = {blade.r_hub:.3f} m,  "
        f"B = {blade.B},  sections = {blade.n()}",
        fontsize=12,
    )

    axes[0].plot(r_raw / blade.R, c_raw, "o", ms=5, color="tomato",
                 zorder=5, label="CSV stations")
    axes[0].plot(blade.r / blade.R, blade.chord, "b-", lw=1.5,
                 label=f"Resampled ({blade.n()} sections)")
    axes[0].set_ylabel("Chord [m]")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(r_raw / blade.R, t_raw, "o", ms=5, color="tomato",
                 zorder=5, label="CSV stations")
    axes[1].plot(blade.r / blade.R, np.degrees(blade.twist), "g-", lw=1.5,
                 label=f"Resampled ({blade.n()} sections)")
    axes[1].set_ylabel("Twist [deg]")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    unique_af = list(dict.fromkeys(af_raw))
    af_to_y   = {name: i for i, name in enumerate(unique_af)}
    y_vals    = np.array([af_to_y[n] for n in af_raw])
    axes[2].step(r_raw / blade.R, y_vals, where="post", color="purple", lw=1.5)
    axes[2].scatter(r_raw / blade.R, y_vals, color="purple", s=20, zorder=5)
    axes[2].set_yticks(list(af_to_y.values()))
    axes[2].set_yticklabels(list(af_to_y.keys()))
    axes[2].set_ylabel("Airfoil")
    axes[2].set_xlabel("r/R")
    axes[2].set_xlim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "blade_geometry.png")


def plot_polars(aero, apply_rotational_correction, case_name):
    """Plot measured, uncorrected, and corrected polar diagnostics."""
    for name, polar in aero.polars_uncorrected_360.items():
        raw = aero.polars_raw[name]
        corrected_reference = aero.polars_corrected_reference.get(name)
        representative = aero.get_representative_section_polars(name)
        a_min = raw["alpha"].min() - 2.0
        a_max = raw["alpha"].max() + 2.0

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f"Polar - {name}   [{case_name}]", fontsize=13)

        for col, (coef, ylabel) in enumerate(zip(["cl", "cd"], ["Cl", "Cd"])):
            ax = axes[0, col]
            ax.plot(polar["alpha"], polar[coef], lw=1.5, label="Uncorrected 360 polar")
            if corrected_reference is not None:
                ax.plot(
                    corrected_reference["alpha"], corrected_reference[coef],
                    lw=1.5, ls="--", label="Corrected reference polar",
                )
            for rr, section_polar in representative.items():
                ax.plot(
                    section_polar["alpha"], section_polar[coef],
                    lw=1.0, alpha=0.8, label=f"Section corrected r/R={rr:.2f}",
                )
            ax.axvspan(a_min, a_max, color="orange", alpha=0.08, label="Measured range")
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel("alpha [deg]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} - full 360 deg")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            ax = axes[1, col]
            ax.plot(raw["alpha"], raw[coef], "o", ms=5, color="tomato",
                    zorder=5, label="Measured 2D polar")
            ax.plot(polar["alpha"], polar[coef], lw=1.5, label="Uncorrected 360 polar")
            if corrected_reference is not None:
                ax.plot(
                    corrected_reference["alpha"], corrected_reference[coef],
                    lw=1.5, ls="--", label="Corrected reference polar",
                )
            for rr, section_polar in representative.items():
                ax.plot(
                    section_polar["alpha"], section_polar[coef],
                    lw=1.0, alpha=0.8, label=f"Section corrected r/R={rr:.2f}",
                )
            ax.set_xlim(a_min, a_max)
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.axvline(0, color="k", lw=0.5, ls="--")
            ax.set_xlabel("alpha [deg]")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} - measured range")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_fig(fig, f"polar_{name}.png")


def plot_cp_surface(cp_raw, lambda_values, pitch_values_deg,
                    opt, active, blade_R, v_surface, rpm_max,
                    cp_min_plot, case_name):
    """Plot 3D Cp(lambda, beta) surface with RPM cap line. Saves cp_surface.png."""
    lambda_grid, beta_grid = np.meshgrid(lambda_values, pitch_values_deg, indexing="ij")
    cp_plot = np.where(cp_raw < cp_min_plot, np.nan, cp_raw)

    omega_max     = rpm_max * 2.0 * np.pi / 60.0
    lambda_at_cap = omega_max * blade_R / v_surface

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
        ax.plot(pitch_values_deg,
                np.full_like(pitch_values_deg, lambda_at_cap), cp_at_cap,
                color="black", lw=2.5, ls="--",
                label=(f"RPM cap = {rpm_max:.0f}: lambda <= {lambda_at_cap:.2f} "
                       f"(at V={v_surface:.0f} m/s)"))

    ax.scatter([opt["pitch"]], [opt["lambda"]], [opt["cp"]],
               color="red", s=80, zorder=5,
               label=f"Optimum  Cp={opt['cp']:.3f}")
    if active["label"] == "User point":
        ax.scatter([active["pitch"]], [active["lambda"]], [active["cp"]],
                   color="orange", s=100, marker="^", zorder=6,
                   label=f"User point  Cp={active['cp']:.3f}")

    ax.set_xlabel("beta [deg]")
    ax.set_ylabel("lambda")
    ax.set_zlabel("Cp")
    ax.set_title(f"Cp(lambda, beta) - {case_name}")
    ax.view_init(elev=25, azim=-135)
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Cp")
    plt.tight_layout()
    save_fig(fig, "cp_surface.png")


def plot_startup_offdesign_cp_surface(cp_offdesign, lambda_values, pitch_values_deg,
                                      case_name, suffix=""):
    """Plot startup / transition / off-design Cp surface."""
    lambda_grid, beta_grid = np.meshgrid(lambda_values, pitch_values_deg, indexing="ij")
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.5, 1.5, 0.8])
    surf = ax.plot_surface(
        beta_grid, lambda_grid, cp_offdesign,
        cmap="coolwarm", edgecolor="none", alpha=0.9,
    )
    ax.set_xlabel("beta [deg]")
    ax.set_ylabel("lambda")
    ax.set_zlabel("Cp")
    ax.set_title(
        "Startup / transition / off-design - not official power-production Cp map\n"
        f"{case_name}"
    )
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Cp")
    plt.tight_layout()
    suffix_part = f"_{suffix}" if suffix else ""
    save_fig(fig, f"cp_surface{suffix_part}_startup_offdesign.png")


def plot_operating_classification_mask(mask, lambda_values, pitch_values_deg,
                                       case_name):
    """Plot operating-envelope interpretation mask."""
    lambda_grid, beta_grid = np.meshgrid(lambda_values, pitch_values_deg, indexing="ij")
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = ListedColormap(["#2ca02c", "#1f77b4", "#ffbf00", "#d62728", "#9467bd"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    mesh = ax.pcolormesh(
        lambda_grid, beta_grid, mask,
        cmap=cmap, norm=norm, shading="nearest",
    )
    ax.set_xlabel("lambda")
    ax.set_ylabel("beta [deg]")
    ax.set_title(f"Operating-envelope classification - {case_name}")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels([
        "valid production",
        "off-design converged",
        "non-converged",
        "negative Cp / torque",
        "high a / K / CT_local",
    ])
    plt.tight_layout()
    save_fig(fig, "cp_surface_operating_classification_mask.png")


def plot_interpolated_cp_surface(cp_visual, interpolation_mask,
                                 lambda_values, pitch_values_deg,
                                 buhl_transition_model, case_name,
                                 suffix="", cp_vmin=0.0, cp_vmax=None):
    """
    Plot zero-filled visualization-only interpolated Cp surface and mask.

    Interpolated Cp is not used for official optimum selection, loads,
    power curve, or AEP.
    """
    suffix_part = f"_{suffix}" if suffix else ""
    label = transition_plot_label(buhl_transition_model)
    title = "Interpolated Cp surface, zero-filled for visualization only"
    cp_vmin = max(0.0, cp_vmin)
    cp_plot = cp_visual

    lambda_grid, beta_grid = np.meshgrid(
        lambda_values, pitch_values_deg, indexing="ij"
    )
    finite_cp = cp_plot[np.isfinite(cp_plot)]
    if cp_vmax is None:
        cp_vmax = float(np.nanmax(finite_cp)) if finite_cp.size else 1.0

    # 1. Interpolated 3D surface
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.5, 1.5, 0.8])
    surf = ax.plot_surface(
        beta_grid, lambda_grid, cp_plot,
        cmap="viridis", edgecolor="none", alpha=0.9,
        vmin=cp_vmin, vmax=cp_vmax,
    )
    ax.set_xlabel("beta [deg]")
    ax.set_ylabel("lambda")
    ax.set_zlabel("Cp")
    ax.set_zlim(cp_vmin, cp_vmax)
    ax.set_title(f"{title}\n{case_name}")
    ax.view_init(elev=25, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Cp")
    plt.tight_layout()
    save_fig(fig, f"cp_surface{suffix_part}_interpolated_3d.png")

    # 2. Interpolated contour
    fig, ax = plt.subplots(figsize=(10, 7))
    levels = np.linspace(cp_vmin, cp_vmax, 30)
    contour = ax.contourf(
        lambda_grid, beta_grid, cp_plot,
        levels=levels, cmap="viridis", vmin=cp_vmin, vmax=cp_vmax,
    )
    ax.set_xlabel("lambda")
    ax.set_ylabel("beta [deg]")
    ax.set_title(f"{title} contour\n{case_name}")
    ax.grid(True, alpha=0.25)
    fig.colorbar(contour, ax=ax, label="Cp")
    plt.tight_layout()
    save_fig(fig, f"cp_surface{suffix_part}_interpolated_contour.png")

    # 3. Interpolation mask
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = ListedColormap(["#1f77b4", "#ffbf00", "#d9d9d9", "#d62728"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    mesh = ax.pcolormesh(
        lambda_grid, beta_grid, interpolation_mask,
        cmap=cmap, norm=norm, shading="nearest",
    )
    ax.set_xlabel("lambda")
    ax.set_ylabel("beta [deg]")
    ax.set_title(
        f"{label} validity / interpolation mask\n"
        "0 = original valid, 1 = interpolated, "
        "2 = missing zero-fill, 3 = negative zero-fill"
    )
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "0 original valid",
        "1 interpolated",
        "2 missing zero-fill",
        "3 negative zero-fill",
    ])
    plt.tight_layout()
    save_fig(fig, f"cp_surface{suffix_part}_interpolation_mask.png")


def plot_cp_ct_curves(cp_curve_df, active, case_name):
    """Plot Cp and Ct vs lambda at the design pitch. Saves cp_ct_curves.png."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Aerodynamic coefficients - {case_name}   "
                 f"beta = {active['pitch']:.2f} deg", fontsize=13)

    ax = axes[0]
    ax.plot(cp_curve_df["lambda"], cp_curve_df["Cp"], "b-o", ms=4, lw=1.5)
    ax.axhline(16/27, color="k", lw=0.8, ls="--", label="Betz limit (0.593)")
    ax.axvline(active["lambda"], color="r", lw=0.8, ls="--",
               label=f"lambda = {active['lambda']:.2f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lambda")
    ax.set_ylabel("Cp")
    ax.set_title("Cp vs lambda")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(cp_curve_df["lambda"], cp_curve_df["Ct"], "g-o", ms=4, lw=1.5)
    ax.axvline(active["lambda"], color="r", lw=0.8, ls="--",
               label=f"lambda = {active['lambda']:.2f}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Ct")
    ax.set_title("Ct vs lambda")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "cp_ct_curves.png")


def plot_cp_pitch_family(cp_pitch_sweep, case_name):
    """Plot Cp/Ct family across pitch angles. Saves cp_ct_pitch_family.png."""
    if not cp_pitch_sweep:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Cp & Ct vs lambda - pitch family - {case_name}", fontsize=13)
    n_curves = len(cp_pitch_sweep)
    colors   = [plt.cm.viridis(i / max(n_curves - 1, 1)) for i in range(n_curves)]

    for ax, key, ylabel, title in [
        (axes[0], "Cp", "Cp", "Cp vs lambda"),
        (axes[1], "Ct", "Ct", "Ct vs lambda"),
    ]:
        for color, (beta, df) in zip(colors, cp_pitch_sweep.items()):
            ax.plot(df["lambda"], df[key], "-o", color=color, lw=1.5,
                    ms=3, label=f"beta = {beta:.1f} deg")
        if key == "Cp":
            ax.axhline(16/27, color="k", lw=0.8, ls="--", label="Betz limit (0.593)")
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "cp_ct_pitch_family.png")


def plot_startup_offdesign_pitch_family(cp_pitch_sweep, case_name):
    """Plot startup / transition / off-design Cp/Ct pitch family."""
    if not cp_pitch_sweep:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Startup / transition / off-design - not official power-production Cp map\n"
        f"{case_name}",
        fontsize=13,
    )
    n_curves = len(cp_pitch_sweep)
    colors = [plt.cm.coolwarm(i / max(n_curves - 1, 1)) for i in range(n_curves)]
    for ax, key, ylabel, title in [
        (axes[0], "Cp", "Cp", "Cp vs lambda"),
        (axes[1], "Ct", "Ct", "Ct vs lambda"),
    ]:
        for color, (beta, df) in zip(colors, cp_pitch_sweep.items()):
            ax.plot(df["lambda"], df[key], "-o", color=color, lw=1.5,
                    ms=3, label=f"beta = {beta:.1f} deg")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "cp_ct_pitch_family_startup_offdesign.png")


def plot_power_thrust(power_df, controller, generator_p_rated_W,
                      eta_drivetrain_at_rated, v_rated, case_name):
    """Plot aerodynamic and electrical power curves plus thrust. Saves power_thrust_curve.png."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Power & thrust curve - {case_name}", fontsize=13)

    ax = axes[0]
    ax.plot(power_df["v_inf"], power_df["power_W"] / 1e3,
            "b-o", ms=5, lw=1.8, label="Aerodynamic (rotor shaft)", zorder=5)
    ax.axhline(controller.p_rated / 1e3, color="b", lw=0.8, ls=":",
               label=f"P_rated aero = {controller.p_rated/1e3:.1f} kW")

    if generator_p_rated_W is not None:
        ax.plot(power_df["v_inf"], power_df["power_elec_W"] / 1e3,
                "r-s", ms=5, lw=1.8, label="Electrical (after drivetrain)", zorder=4)
        ax.axhline(generator_p_rated_W / 1e3, color="r", lw=0.8, ls=":",
                   label=f"P_rated elec = {generator_p_rated_W/1e3:.1f} kW (nameplate)")
        ax.fill_between(power_df["v_inf"],
                        power_df["power_elec_W"] / 1e3, power_df["power_W"] / 1e3,
                        color="black", alpha=0.10, hatch="//", lw=0,
                        label="Drivetrain losses")
        v_at_nameplate = first_crossing(
            power_df["v_inf"].values, power_df["power_elec_W"].values, generator_p_rated_W
        )
        if v_at_nameplate is not None:
            ax.scatter([v_at_nameplate], [generator_p_rated_W / 1e3],
                       color="magenta", s=120, marker="*", zorder=10,
                       label=f"Nameplate reached @ V = {v_at_nameplate:.2f} m/s")

    ax.axvline(v_rated, color="grey", lw=0.8, ls="--", label=f"V_rated = {v_rated} m/s")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Power curve P(V)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(power_df["v_inf"], power_df["thrust_N"], "g-o", ms=5, lw=1.8)
    ax.axvline(v_rated, color="grey", lw=0.8, ls="--", label=f"V_rated = {v_rated} m/s")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title("Thrust curve T(V)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "power_thrust_curve.png")


def plot_drivetrain_efficiency(power_df, eta_drivetrain_at_rated, v_rated, case_name):
    """Plot generator efficiency vs wind speed. Saves drivetrain_efficiency.png."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Drivetrain efficiency - {case_name}", fontsize=13)
    ax.plot(power_df["v_inf"], 100.0 * power_df["eta_drivetrain"], "k-o", ms=5, lw=1.5)
    ax.axhline(100.0 * eta_drivetrain_at_rated, color="r", lw=0.8, ls="--",
               label=f"Target at rated: {100*eta_drivetrain_at_rated:.0f}%")
    ax.axvline(v_rated, color="grey", lw=0.8, ls="--", label=f"V_rated = {v_rated} m/s")
    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Drivetrain efficiency [%]")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "drivetrain_efficiency.png")


def plot_loads_curve(loads_df, controller, pitch_loads_deg, case_name):
    """Plot fixed-RPM torque and thrust vs wind speed. Saves loads_fixed_rpm.png."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Loads at fixed RPM - {case_name}\n"
        f"RPM = {controller.rpm_rated:.1f},  beta = {pitch_loads_deg} deg",
        fontsize=13,
    )
    axes[0].plot(loads_df["v_inf"], loads_df["torque_Nm"], "b-o", ms=5, lw=1.5)
    axes[0].set_xlabel("Wind speed [m/s]")
    axes[0].set_ylabel("Shaft torque [N*m]")
    axes[0].set_title("Shaft torque")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(loads_df["v_inf"], loads_df["thrust_N"], "b-o", ms=5, lw=1.5)
    axes[1].set_xlabel("Wind speed [m/s]")
    axes[1].set_ylabel("Thrust [N]")
    axes[1].set_title("Rotor thrust")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "loads_fixed_rpm.png")


def plot_section_analysis(sec_df, active, v_surface, case_name):
    """Plot 6-panel radial section analysis. Saves sections_design_point.png."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Radial section analysis - {case_name}   "
        f"lambda = {active['lambda']:.2f},  beta = {active['pitch']:.2f} deg,  "
        f"V = {v_surface} m/s",
        fontsize=12,
    )
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
        ax.set_xlabel("r/R")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        for x in nc["r_R"].values:
            ax.axvline(x, color="orange", lw=0.8, ls="--", alpha=0.5)
    plt.tight_layout()
    save_fig(fig, "sections_design_point.png")


def print_summary(case_name, blade, opt, active, controller, power_df,
                  aep_kwh, aep_elec_kwh, generator_p_rated_W,
                  eta_drivetrain_at_rated, weibull_c, weibull_k, output_dir):
    """Print the final results summary to stdout."""
    print("\n" + "=" * 52)
    print(f"  RESULTS SUMMARY - {case_name}")
    print("=" * 52)
    print(f"  R               = {blade.R:.4f} m   (B = {blade.B})")
    print(f"  --- Aerodynamic optimum ---")
    print(f"  lambda_opt      = {opt['lambda']:.3f}")
    print(f"  beta_opt        = {opt['pitch']:.2f} deg")
    print(f"  Cp_opt          = {opt['cp']:.4f}")
    if active["label"] == "User point":
        print(f"  --- User-defined active point ---")
        print(f"  lambda_user     = {active['lambda']:.3f}")
        print(f"  beta_user       = {active['pitch']:.2f} deg")
        print(f"  Cp_user         = {active['cp']:.4f}  "
              f"(delta = {active['cp'] - opt['cp']:+.4f})")
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
        print(f"  AEP loss to drivetrain = "
              f"{aep_kwh - aep_elec_kwh:.1f} kWh/year "
              f"({100 * (aep_kwh - aep_elec_kwh) / aep_kwh:.1f}%)")
        v_at_nameplate = first_crossing(
            power_df["v_inf"].values, power_df["power_elec_W"].values, generator_p_rated_W
        )
        if v_at_nameplate is not None:
            print(f"  Nameplate ({generator_p_rated_W:.0f} W) "
                  f"reached at V = {v_at_nameplate:.2f} m/s")
        else:
            print(f"  Nameplate ({generator_p_rated_W:.0f} W) never reached "
                  f"(peak electrical = {p_elec_max:.1f} W)")
    print("=" * 52)
    print(f"\nOutputs in: {output_dir}")


def print_aep_comparison(aep_raw_kwh, aep_filtered_kwh,
                         aep_raw_elec_kwh, aep_filtered_elec_kwh,
                         used_filtered_interp, used_filtered_elec_interp):
    """Report raw and filtered AEP explicitly."""
    print("\n  --- AEP validity ---")
    print(f"  AEP raw aero      = {aep_raw_kwh:.1f} kWh/year")
    print(f"  AEP filtered aero = {aep_filtered_kwh:.1f} kWh/year")
    if used_filtered_interp:
        print("  Warning: filtered aero AEP used interpolation over invalid rows")

    if aep_raw_elec_kwh is not None and aep_filtered_elec_kwh is not None:
        print(f"  AEP raw elec      = {aep_raw_elec_kwh:.1f} kWh/year")
        print(f"  AEP filtered elec = {aep_filtered_elec_kwh:.1f} kWh/year")
        if used_filtered_elec_interp:
            print("  Warning: filtered electrical AEP used interpolation over invalid rows")


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    os.makedirs(output_dir, exist_ok=True)

    if high_induction_model not in ("buhl", "reference_glauert"):
        raise ValueError(
            "high_induction_model must be 'buhl' or 'reference_glauert', "
            f"got {high_induction_model!r}"
        )

    # --------------------------------------------------
    # 1. Blade geometry
    # --------------------------------------------------
    blade = BladeGeometry(
        file_path=geometry_file, n_sections=n_sections,
        B=n_blades, hub_diameter=hub_diameter,
        r_is_span_offset=geometry_r_is_span_offset,
    )
    blade.load_csv()
    blade.Malla()
    print(f"Blade: R = {blade.R:.4f} m, sections = {blade.n()}, B = {blade.B}")
    export_geometry_diagnostics(blade, geometry_file, output_dir)
    plot_blade_geometry(blade, geometry_file, case_name)

    # --------------------------------------------------
    # 2. Airfoil aerodynamics
    # --------------------------------------------------
    aero = Aero(
        geometry=blade, polar_files=polar_files,
        extrapolate_360=extrapolate_360, step_deg=1.0,
        blend_half_width_frac=blend_half_width_frac,
    )
    aero.load_polars()
    if apply_rotational_correction:
        rot_corr_diagnostics = aero.correct_polars(
            tsr_ref=float(np.median(lambda_values)),
            v_ref=v_surface,
            min_r_over_R=min_r_over_R_for_rot_corr,
            max_r_over_R=max_r_over_R_for_rot_corr,
            alpha_min_deg=alpha_min_rot_corr_deg,
            alpha_max_deg=alpha_max_rot_corr_deg,
            delta_cl_max=delta_cl_max,
            apply_before_viterna=apply_rot_corr_before_viterna,
        )
        if rot_corr_diagnostics:
            rot_diag_df = pd.DataFrame(rot_corr_diagnostics)
            rot_diag_df.to_csv(
                os.path.join(output_dir, "rotational_correction_diagnostics.csv"),
                index=False,
            )
            warning_rows = rot_diag_df[
                rot_diag_df["post_stall_or_extrapolated_points_affected"] > 0
            ]
            for _, row in warning_rows.iterrows():
                print(
                    "  Warning: rotational correction affected "
                    f"{int(row['post_stall_or_extrapolated_points_affected'])} "
                    "post-stall/extrapolated polar points for "
                    f"{row['airfoil']}"
                )
    plot_polars(aero, apply_rotational_correction, case_name)

    # --------------------------------------------------
    # 3. BEM solver
    # --------------------------------------------------
    solver = BEMSolver(
        geometry=blade, aero=aero, rho=rho,
        max_iter=max_iter, tol=tol, relaxation=relaxation,
        use_tip_loss=use_tip_loss, use_root_loss=use_root_loss,
        use_glauert=use_glauert, use_root_cutoff=use_root_cutoff,
        root_cutoff_fraction=root_cutoff_frac,
        high_induction_model=high_induction_model,
        record_iteration_history=False,
        buhl_transition_model=buhl_transition_model,
        buhl_deactivation_count_required=buhl_deactivation_count_required,
    )

    # --------------------------------------------------
    # 4. Cp(lambda, beta) surface
    # --------------------------------------------------
    print(f"\nRunning Cp(lambda, beta) surface "
          f"({len(lambda_values)} x {len(pitch_values_deg)} points)...")
    cp_raw, cp_surface_summary_df, cp_surface_sections_df = run_cp_surface(
        solver, blade, lambda_values, pitch_values_deg, v_surface
    )

    cp_filtered = build_filtered_cp_surface(
        cp_surface_summary_df, lambda_values, pitch_values_deg, high_induction_model
    )

    save_cp_surface_outputs(
        cp_raw, cp_filtered, cp_surface_summary_df, cp_surface_sections_df,
        lambda_values, pitch_values_deg, output_dir,
    )

    cp_validated, cp_surface_validated_df = build_validated_cp_surface(
        cp_surface_summary_df, cp_surface_sections_df,
        lambda_values, pitch_values_deg,
        max_a_valid, max_K_valid, max_CT_local_valid,
    )
    save_validated_cp_surface(
        cp_validated, cp_surface_validated_df,
        lambda_values, pitch_values_deg, output_dir,
    )
    cp_startup_offdesign = build_startup_offdesign_cp_surface(
        cp_surface_summary_df, lambda_values, pitch_values_deg
    )
    save_startup_offdesign_cp_surface(
        cp_startup_offdesign, lambda_values, pitch_values_deg, output_dir,
    )
    (
        cp_startup_offdesign_interpolated,
        cp_startup_offdesign_interpolation_mask,
    ) = build_startup_offdesign_interpolated_cp_surface(
        cp_startup_offdesign, lambda_values, pitch_values_deg
    )
    save_startup_offdesign_interpolated_cp_surface(
        cp_startup_offdesign_interpolated,
        cp_startup_offdesign_interpolation_mask,
        lambda_values, pitch_values_deg, output_dir,
    )
    operating_classification_mask = build_operating_classification_mask(
        cp_surface_validated_df, lambda_values, pitch_values_deg
    )
    save_operating_classification_mask(
        operating_classification_mask, lambda_values, pitch_values_deg, output_dir,
    )

    cp_interpolated, cp_interpolation_mask, cp_interpolation_stats = (
        build_interpolated_cp_surface(
            cp_validated, lambda_values, pitch_values_deg
        )
    )
    save_interpolated_cp_surface(
        cp_interpolated, cp_interpolation_mask,
        lambda_values, pitch_values_deg, output_dir,
    )
    (
        cp_interpolated_visual,
        cp_interpolation_visual_mask,
        cp_interpolation_visual_stats,
    ) = build_interpolated_cp_visual_surface(
        cp_interpolated, cp_interpolation_mask
    )
    save_interpolated_cp_visual_surface(
        cp_interpolated_visual, cp_interpolation_visual_mask,
        lambda_values, pitch_values_deg, output_dir,
    )
    plot_interpolated_cp_surface(
        cp_interpolated_visual, cp_interpolation_visual_mask,
        lambda_values, pitch_values_deg,
        buhl_transition_model, case_name,
        cp_vmin=0.0,
    )
    print(
        "  Interpolated Cp visualization sources: "
        f"{cp_interpolation_stats['valid_nonnegative_sources']} "
        "valid non-negative points"
    )
    print(
        "  Interpolated Cp visualization remaining NaNs: "
        f"{cp_interpolation_stats['remaining_nan']}"
    )
    print(
        "  Negative Cp region excluded from interpolation: "
        f"{cp_interpolation_stats['negative_region_excluded']} "
        f"({cp_interpolation_stats['excluded_negative_points']} points)"
    )
    print(
        "  Zero-filled interpolated visualization points: "
        f"{cp_interpolation_visual_stats['zero_filled_nan_points']} NaN, "
        f"{cp_interpolation_visual_stats['zero_filled_negative_points']} negative"
    )
    print(
        "  Warning: This zero-filled interpolated Cp surface is for visualization "
        "only. Zero-filled regions must not be used for optimum selection, "
        "loads, power curve, or AEP."
    )

    if (cp_surface_summary_df["n_nonconverged"] > 0).any():
        n_cases = int((cp_surface_summary_df["n_nonconverged"] > 0).sum())
        print(f"  Warning: {n_cases} lambda-beta cases contain non-converged sections")

    raw_opt, _ = find_design_point(
        cp_raw, lambda_values, pitch_values_deg, None, None
    )
    filtered_opt, active = find_design_point(
        cp_filtered, lambda_values, pitch_values_deg, user_lambda, user_pitch
    )
    validated_opt, active = find_design_point(
        cp_validated, lambda_values, pitch_values_deg, user_lambda, user_pitch
    )
    opt = validated_opt

    print(f"\nRaw optimum:  "
          f"lambda = {raw_opt['lambda']:.3f},  beta = {raw_opt['pitch']:.2f} deg,  "
          f"Cp = {raw_opt['cp']:.4f}")
    print(f"Valid/filtered optimum:  "
          f"lambda = {filtered_opt['lambda']:.3f},  beta = {filtered_opt['pitch']:.2f} deg,  "
          f"Cp = {filtered_opt['cp']:.4f}")
    print(f"Validated official optimum:  "
          f"lambda = {validated_opt['lambda']:.3f},  beta = {validated_opt['pitch']:.2f} deg,  "
          f"Cp = {validated_opt['cp']:.4f}")

    if optima_differ(raw_opt, validated_opt):
        print("  Warning: raw optimum differs from validated official optimum")
        print(f"    raw:      lambda = {raw_opt['lambda']:.3f}, "
              f"beta = {raw_opt['pitch']:.2f} deg, Cp = {raw_opt['cp']:.4f}")
        print(f"    validated: lambda = {validated_opt['lambda']:.3f}, "
              f"beta = {validated_opt['pitch']:.2f} deg, Cp = {validated_opt['cp']:.4f}")

    if run_refined_sweep:
        print(f"\nRunning refined Cp(lambda, beta) surface "
              f"({len(lambda_refined)} x {len(pitch_refined_deg)} points)...")
        cp_refined_raw, refined_summary_df, refined_sections_df = run_cp_surface(
            solver, blade, lambda_refined, pitch_refined_deg, v_surface
        )
        cp_refined_filtered = build_filtered_cp_surface(
            refined_summary_df, lambda_refined, pitch_refined_deg, high_induction_model
        )
        save_cp_surface_outputs(
            cp_refined_raw, cp_refined_filtered,
            refined_summary_df, refined_sections_df,
            lambda_refined, pitch_refined_deg, output_dir,
            suffix="refined",
        )
        cp_refined_validated, refined_validated_df = build_validated_cp_surface(
            refined_summary_df, refined_sections_df,
            lambda_refined, pitch_refined_deg,
            max_a_valid, max_K_valid, max_CT_local_valid,
        )
        save_validated_cp_surface(
            cp_refined_validated, refined_validated_df,
            lambda_refined, pitch_refined_deg, output_dir,
            suffix="refined",
        )
        save_cp_validity_mask_and_stats(
            refined_summary_df, lambda_refined, pitch_refined_deg,
            output_dir, suffix="refined",
        )

        (
            cp_refined_interpolated,
            cp_refined_interpolation_mask,
            cp_refined_interpolation_stats,
        ) = (
            build_interpolated_cp_surface(
                cp_refined_validated, lambda_refined, pitch_refined_deg
            )
        )
        save_interpolated_cp_surface(
            cp_refined_interpolated, cp_refined_interpolation_mask,
            lambda_refined, pitch_refined_deg, output_dir,
            suffix="refined",
        )
        (
            cp_refined_interpolated_visual,
            cp_refined_interpolation_visual_mask,
            cp_refined_interpolation_visual_stats,
        ) = build_interpolated_cp_visual_surface(
            cp_refined_interpolated, cp_refined_interpolation_mask
        )
        save_interpolated_cp_visual_surface(
            cp_refined_interpolated_visual,
            cp_refined_interpolation_visual_mask,
            lambda_refined, pitch_refined_deg, output_dir,
            suffix="refined",
        )
        plot_interpolated_cp_surface(
            cp_refined_interpolated_visual,
            cp_refined_interpolation_visual_mask,
            lambda_refined, pitch_refined_deg,
            buhl_transition_model, case_name,
            suffix="refined", cp_vmin=0.0,
        )
        print(
            "  Refined interpolated Cp visualization sources: "
            f"{cp_refined_interpolation_stats['valid_nonnegative_sources']} "
            "valid non-negative points"
        )
        print(
            "  Refined interpolated Cp visualization remaining NaNs: "
            f"{cp_refined_interpolation_stats['remaining_nan']}"
        )
        print(
            "  Refined negative Cp region excluded from interpolation: "
            f"{cp_refined_interpolation_stats['negative_region_excluded']} "
            f"({cp_refined_interpolation_stats['excluded_negative_points']} points)"
        )
        print(
            "  Refined zero-filled interpolated visualization points: "
            f"{cp_refined_interpolation_visual_stats['zero_filled_nan_points']} NaN, "
            f"{cp_refined_interpolation_visual_stats['zero_filled_negative_points']} negative"
        )
        print(
            "  Warning: This zero-filled interpolated Cp surface is for "
            "visualization only. Zero-filled regions must not be used for "
            "optimum selection, loads, power curve, or AEP."
        )

        refined_raw_opt, _ = find_design_point(
            cp_refined_raw, lambda_refined, pitch_refined_deg, None, None
        )
        refined_filtered_opt, _ = find_design_point(
            cp_refined_filtered, lambda_refined, pitch_refined_deg, None, None
        )
        refined_validated_opt, _ = find_design_point(
            cp_refined_validated, lambda_refined, pitch_refined_deg, None, None
        )
        print(f"Refined raw optimum:  "
              f"lambda = {refined_raw_opt['lambda']:.3f}, "
              f"beta = {refined_raw_opt['pitch']:.2f} deg, "
              f"Cp = {refined_raw_opt['cp']:.4f}")
        print(f"Refined valid/filtered optimum:  "
              f"lambda = {refined_filtered_opt['lambda']:.3f}, "
              f"beta = {refined_filtered_opt['pitch']:.2f} deg, "
              f"Cp = {refined_filtered_opt['cp']:.4f}")
        print(f"Refined validated official optimum:  "
              f"lambda = {refined_validated_opt['lambda']:.3f}, "
              f"beta = {refined_validated_opt['pitch']:.2f} deg, "
              f"Cp = {refined_validated_opt['cp']:.4f}")

    print(f"\nDesign point ({active['label']}):  "
          f"lambda = {active['lambda']:.3f},  beta = {active['pitch']:.2f} deg,  "
          f"Cp = {active['cp']:.4f}")

    plot_cp_surface(
        cp_validated, lambda_values, pitch_values_deg,
        opt, active, blade.R, v_surface, rpm_max, cp_min_plot, case_name,
    )
    plot_startup_offdesign_cp_surface(
        cp_startup_offdesign, lambda_values, pitch_values_deg, case_name
    )
    plot_startup_offdesign_cp_surface(
        cp_startup_offdesign_interpolated,
        lambda_values, pitch_values_deg, case_name, suffix="interpolated",
    )
    plot_operating_classification_mask(
        operating_classification_mask, lambda_values, pitch_values_deg, case_name
    )

    # --------------------------------------------------
    # 5. Cp(lambda) curves
    # --------------------------------------------------
    lambda_curve = np.linspace(lambda_values.min(), lambda_values.max(), 60)

    print(f"\nCp curve at design pitch ({active['pitch']:.2f} deg)...")
    cp_curve_df = run_cp_lambda_sweep(
        solver, blade, lambda_curve, active["pitch"], v_surface, desc="design"
    )
    cp_curve_df.to_csv(os.path.join(output_dir, "cp_curve.csv"), index=False)
    plot_cp_ct_curves(cp_curve_df, active, case_name)

    cp_pitch_sweep = {}
    if pitch_sweep_deg:
        print(f"\nCp curve family ({len(pitch_sweep_deg)} pitch values)...")
        for beta in pitch_sweep_deg:
            cp_pitch_sweep[beta] = run_cp_lambda_sweep(
                solver, blade, lambda_curve, beta, v_surface, desc=f"beta={beta:.0f}"
            )
        combined = pd.DataFrame({"lambda": lambda_curve})
        for beta, df in cp_pitch_sweep.items():
            combined[f"Cp_beta_{beta:g}"] = df["Cp"].values
            combined[f"Ct_beta_{beta:g}"] = df["Ct"].values
            combined[f"n_nonconverged_beta_{beta:g}"] = df["n_nonconverged"].values
            combined[f"valid_beta_{beta:g}"] = df["valid"].values
            combined[f"invalid_reason_beta_{beta:g}"] = df["invalid_reason"].values
        combined.to_csv(os.path.join(output_dir, "cp_curve_pitch_sweep.csv"), index=False)
        startup_offdesign_report = build_startup_offdesign_report(cp_pitch_sweep)
        startup_offdesign_report.to_csv(
            os.path.join(output_dir, "cp_curve_pitch_sweep_startup_offdesign_report.csv"),
            index=False,
        )
    plot_cp_pitch_family(cp_pitch_sweep, case_name)
    plot_startup_offdesign_pitch_family(cp_pitch_sweep, case_name)

    # --------------------------------------------------
    # 6. Power curve & drivetrain
    # --------------------------------------------------
    controller = WindTurbineController(
        solver=solver, lambda_design=active["lambda"], pitch_design=active["pitch"],
        rotor_radius=blade.R, v_rated=v_rated, v_cut_in=v_cut_in,
        v_cut_out=v_cut_out, rpm_max=rpm_max, pitch_control=True,
    )
    print(f"\n{controller}")

    print("\nRunning power curve P(V)...")
    power_df = controller.compute_power_curve(
        wind_speeds=wind_speeds_power, convergence_check_fn=n_nonconverged,
    )
    power_df = add_power_curve_validity(power_df, blade.n())

    if generator_p_rated_W is not None:
        apply_drivetrain(power_df, generator_p_rated_W,
                         eta_drivetrain_at_rated, generator_const_loss_frac)

    power_filtered_df = build_filtered_power_curve(power_df)
    print_invalid_power_curve_rows(power_df)
    if getattr(controller, "p_rated_non_converged", 0) > 0:
        print(
            "  Warning: controller rated-power reference used "
            f"{controller.p_rated_non_converged} non-converged sections"
        )

    power_df.to_csv(os.path.join(output_dir, "power_curve_raw.csv"), index=False)
    power_df.to_csv(os.path.join(output_dir, "power_curve.csv"), index=False)
    power_filtered_df.to_csv(
        os.path.join(output_dir, "power_curve_filtered.csv"), index=False
    )

    aep_kwh = controller.compute_aep(power_df, weibull_c, weibull_k)
    power_aep_df, used_filtered_interp = interpolate_power_curve_for_aep(
        power_filtered_df, "power_W"
    )
    aep_filtered_kwh = controller.compute_aep(power_aep_df, weibull_c, weibull_k)

    aep_elec_kwh = None
    aep_filtered_elec_kwh = None
    used_filtered_elec_interp = False
    if generator_p_rated_W is not None:
        aep_elec_kwh = compute_aep_electrical(power_df, weibull_c, weibull_k)
        power_aep_elec_df, used_filtered_elec_interp = interpolate_power_curve_for_aep(
            power_filtered_df, "power_elec_W"
        )
        aep_filtered_elec_kwh = compute_aep_electrical(
            power_aep_elec_df, weibull_c, weibull_k
        )

    if used_filtered_interp or used_filtered_elec_interp:
        power_aep_export_df = power_aep_df.copy()
        if generator_p_rated_W is not None:
            power_aep_export_df["power_elec_W"] = power_aep_elec_df["power_elec_W"]
            power_aep_export_df["power_elec_W_interpolated_for_aep"] = (
                power_aep_elec_df["power_elec_W_interpolated_for_aep"]
            )
        power_aep_export_df["interpolated_for_aep"] = (
            power_aep_export_df.get("power_W_interpolated_for_aep", False)
        )
        if "power_elec_W_interpolated_for_aep" in power_aep_export_df.columns:
            power_aep_export_df["interpolated_for_aep"] = (
                power_aep_export_df["interpolated_for_aep"]
                | power_aep_export_df["power_elec_W_interpolated_for_aep"]
            )
        power_aep_export_df.to_csv(
            os.path.join(output_dir, "power_curve_aep_interpolated.csv"),
            index=False,
        )
        print(
            "  Warning: AEP depends on interpolated values over non-converged "
            "power-curve rows. See power_curve_aep_interpolated.csv."
        )

    print_aep_comparison(
        aep_kwh, aep_filtered_kwh,
        aep_elec_kwh, aep_filtered_elec_kwh,
        used_filtered_interp, used_filtered_elec_interp,
    )

    plot_power_thrust(power_df, controller, generator_p_rated_W,
                      eta_drivetrain_at_rated, v_rated, case_name)
    if generator_p_rated_W is not None:
        plot_drivetrain_efficiency(power_df, eta_drivetrain_at_rated, v_rated, case_name)

    # --------------------------------------------------
    # 7. Loads at fixed RPM
    # --------------------------------------------------
    print(f"\nLoads curve at RPM = {controller.rpm_rated:.1f}, "
          f"beta = {pitch_loads_deg} deg...")
    loads_df = run_loads_curve(
        solver, wind_speeds_loads, controller.omega_rated, pitch_loads_deg
    )
    loads_df.to_csv(os.path.join(output_dir, "loads_curve.csv"), index=False)
    plot_loads_curve(loads_df, controller, pitch_loads_deg, case_name)

    # --------------------------------------------------
    # 8. Section analysis at design point
    # --------------------------------------------------
    print("\nSection analysis at design point...")
    sec_df = run_section_analysis(
        solver, blade, active["lambda"], active["pitch"], v_surface
    )
    sec_df.to_csv(os.path.join(output_dir, "sections_design_point.csv"), index=False)
    n_nc = (~sec_df["converged"]).sum()
    if n_nc > 0:
        print(f"  Note: {n_nc} non-converged sections")
    plot_section_analysis(sec_df, active, v_surface, case_name)

    if save_iteration_history:
        print("\nSaving selected iteration-history diagnostics...")
        hist_df = run_iteration_history_diagnostics(
            solver=solver,
            blade=blade,
            cases=iteration_history_cases,
            v_surface=v_surface,
            rR_min=iteration_history_rR_min,
            rR_max=iteration_history_rR_max,
        )
        hist_path = os.path.join(output_dir, "iteration_history_diagnostics.csv")
        hist_df.to_csv(hist_path, index=False)
        print(f"  Saved {len(hist_df)} iteration rows to {hist_path}")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print_summary(
        case_name, blade, opt, active, controller, power_df,
        aep_kwh, aep_elec_kwh, generator_p_rated_W,
        eta_drivetrain_at_rated, weibull_c, weibull_k, output_dir,
    )
