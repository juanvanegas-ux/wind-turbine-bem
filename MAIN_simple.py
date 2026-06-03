"""
MAIN_simple.py

Compact visualization / analysis driver for:
  1. Blade geometry
  2. Airfoil polars
  3. Cp(lambda, beta) surfaces
  4. Fixed-wind-speed pitch sweeps

This script intentionally omits controller logic, AEP, wind-speed power curves,
large debug exports, and batch load-distribution workflows.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.interpolate import griddata
from tqdm import tqdm

from Geometria import BladeGeometry
from Aero import Aero
from BEM import BEMSolver


# ==================================================
# CONFIGURATION
# ==================================================

case_name = "SIMPLE_RUN"
output_dir = f"Outputs/{case_name}"

# --- Geometry ---
geometry_file = "Inputs/smart/bladegeom_smart.csv"
polar_files = {
    "cylinder": "Inputs/smart/cylinder.csv",
    "S822": "Inputs/smart/S822.csv",
    "S823": "Inputs/smart/S823.csv",
}
geometry_r_is_span_offset = True
n_sections = 50
n_blades = 3
hub_diameter = 0.2

# --- Aerodynamics / solver ---
rho = 1.225
extrapolate_360 = True
apply_rotational_correction = True
min_r_over_R_for_rot_corr = 0.15
max_r_over_R_for_rot_corr = 0.45
alpha_min_rot_corr_deg = -5.0
alpha_max_rot_corr_deg = 20.0
delta_cl_max = 0.2
apply_rot_corr_before_viterna = True
high_induction_model = "buhl"
buhl_transition_model = "hard"
buhl_deactivation_count_required = 3
use_tip_loss = True
use_root_loss = True
use_glauert = True
use_root_cutoff = False
root_cutoff_fraction = 0.20
max_iter = 500
tol = 1e-3
relaxation = 0.15

# --- Cp surface ---
v_surface = 8.0
lambda_values = np.linspace(0.0, 10, 40)
pitch_values_deg = np.linspace(0, 15, 40)

# --- Pitch sweep ---
pitch_sweep_deg = [-3, -2,-1, 0, 1, 2, 3, 4, 5]
lambda_sweep_values = np.linspace(0.0,10, 60)
pitch_sweep_vinf = 8.0

# --- Plotting / validation ---
create_filtered_surface = True
create_interpolated_surface = True
interpolation_visual_zero_fill = False
cp_min_plot = 0.0
save_plots = True


# ==================================================
# HELPERS
# ==================================================

def make_output_dir(path):
    """Create output folder if needed."""
    os.makedirs(path, exist_ok=True)


def save_figure(fig, filename):
    """Save and close one figure when plotting is enabled."""
    if save_plots:
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_blade_and_aero():
    """Load geometry, aerodynamics, and build the BEM solver."""
    if high_induction_model not in ("buhl", "reference_glauert"):
        raise ValueError(
            "high_induction_model must be 'buhl' or 'reference_glauert', "
            f"got {high_induction_model!r}"
        )

    blade = BladeGeometry(
        file_path=geometry_file,
        n_sections=n_sections,
        B=n_blades,
        hub_diameter=hub_diameter,
        r_is_span_offset=geometry_r_is_span_offset,
    )
    blade.load_csv()
    blade.Malla()

    aero = Aero(
        geometry=blade,
        polar_files=polar_files,
        extrapolate_360=extrapolate_360,
        step_deg=1.0,
    )
    aero.load_polars()
    if apply_rotational_correction:
        aero.correct_polars(
            tsr_ref=float(np.median(lambda_values)),
            v_ref=v_surface,
            min_r_over_R=min_r_over_R_for_rot_corr,
            max_r_over_R=max_r_over_R_for_rot_corr,
            alpha_min_deg=alpha_min_rot_corr_deg,
            alpha_max_deg=alpha_max_rot_corr_deg,
            delta_cl_max=delta_cl_max,
            apply_before_viterna=apply_rot_corr_before_viterna,
        )

    solver = BEMSolver(
        geometry=blade,
        aero=aero,
        rho=rho,
        max_iter=max_iter,
        tol=tol,
        relaxation=relaxation,
        use_tip_loss=use_tip_loss,
        use_root_loss=use_root_loss,
        use_glauert=use_glauert,
        use_root_cutoff=use_root_cutoff,
        root_cutoff_fraction=root_cutoff_fraction,
        high_induction_model=high_induction_model,
        buhl_transition_model=buhl_transition_model,
        buhl_deactivation_count_required=buhl_deactivation_count_required,
    )
    return blade, aero, solver


def plot_geometry(blade):
    """Plot chord, twist, and airfoil distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    r_over_R = blade.r / blade.R
    axes[0].plot(r_over_R, blade.chord, "b-o", ms=3)
    axes[0].set_ylabel("chord [m]")
    axes[1].plot(r_over_R, np.degrees(blade.twist), "g-o", ms=3)
    axes[1].set_ylabel("twist [deg]")

    names = list(dict.fromkeys(blade.airfoil_names))
    name_to_y = {name: i for i, name in enumerate(names)}
    y = [name_to_y[name] for name in blade.airfoil_names]
    axes[2].step(blade.airfoil_r / blade.R, y, where="post")
    axes[2].scatter(blade.airfoil_r / blade.R, y, s=20)
    axes[2].set_yticks(list(name_to_y.values()))
    axes[2].set_yticklabels(list(name_to_y.keys()))
    axes[2].set_ylabel("airfoil")
    axes[2].set_xlabel("r/R")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "geometry_plot.png")

    pd.DataFrame({
        "r": blade.r,
        "r_R": r_over_R,
        "chord": blade.chord,
        "twist_deg": np.degrees(blade.twist),
    }).to_csv(os.path.join(output_dir, "geometry_summary.csv"), index=False)


def plot_airfoil_polars(aero):
    """Plot measured, uncorrected, and corrected polar diagnostics."""
    for name, polar in aero.polars_uncorrected_360.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        measured = aero.polars_raw[name]
        corrected_reference = aero.polars_corrected_reference.get(name)
        representative = aero.get_representative_section_polars(name)
        panels = [
            ("cl", "Cl"),
            ("cd", "Cd"),
            (None, "Cl/Cd"),
        ]
        for ax, (col, ylabel) in zip(axes, panels):
            if col is None:
                ax.plot(
                    measured["alpha"],
                    measured["cl"] / measured["cd"].replace(0, np.nan),
                    "o",
                    ms=4,
                    label="measured 2D",
                )
                ax.plot(
                    polar["alpha"],
                    polar["cl"] / polar["cd"].replace(0, np.nan),
                    label="uncorrected 360",
                )
                if corrected_reference is not None:
                    ax.plot(
                        corrected_reference["alpha"],
                        corrected_reference["cl"] / corrected_reference["cd"].replace(0, np.nan),
                        "--",
                        label="corrected reference",
                    )
                for rr, section_polar in representative.items():
                    ax.plot(
                        section_polar["alpha"],
                        section_polar["cl"] / section_polar["cd"].replace(0, np.nan),
                        label=f"section r/R={rr:.2f}",
                    )
            else:
                ax.plot(measured["alpha"], measured[col], "o", ms=4, label="measured 2D")
                ax.plot(polar["alpha"], polar[col], label="uncorrected 360")
                if corrected_reference is not None:
                    ax.plot(
                        corrected_reference["alpha"], corrected_reference[col],
                        "--", label="corrected reference",
                    )
                for rr, section_polar in representative.items():
                    ax.plot(
                        section_polar["alpha"], section_polar[col],
                        label=f"section r/R={rr:.2f}",
                    )
            ax.set_xlabel("alpha [deg]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"Airfoil polar - {name}")
        fig.tight_layout()
        save_figure(fig, f"airfoil_{name}_polar_plot.png")


def summarize_rotor_result(result):
    """Return compact rotor diagnostics for one lambda-pitch case."""
    sections = result["sections"]
    n_nonconverged = sum(not s.get("converged", False) for s in sections)
    n_physically_invalid = sum(not s.get("physically_valid", True) for s in sections)
    max_k_values = [s.get("K", np.nan) for s in sections]
    max_a_values = [s.get("a", np.nan) for s in sections]
    return {
        "Cp": result["Cp"],
        "Ct": result["Ct"],
        "torque": result["torque"],
        "thrust": result["thrust"],
        "n_nonconverged": n_nonconverged,
        "n_physically_invalid": n_physically_invalid,
        "max_K": np.nanmax(max_k_values),
        "max_a": np.nanmax(max_a_values),
    }


def run_cp_surface(solver, blade):
    """Compute raw Cp(lambda, pitch) surface and summary table."""
    cp_raw = np.full((len(lambda_values), len(pitch_values_deg)), np.nan)
    rows = []
    for i, lam in enumerate(tqdm(lambda_values, desc="Cp surface lambda sweep")):
        omega = lam * v_surface / blade.R
        for j, pitch_deg in enumerate(pitch_values_deg):
            result = solver.solve_rotor(
                V_inf=v_surface,
                omega=omega,
                pitch=np.radians(pitch_deg),
            )
            summary = summarize_rotor_result(result)
            cp_raw[i, j] = summary["Cp"]
            rows.append({
                "lambda": lam,
                "pitch_deg": pitch_deg,
                **summary,
            })
    return cp_raw, pd.DataFrame(rows)


def build_filtered_surface(summary_df):
    """Keep only converged / currently valid Cp values."""
    cp_filtered = np.full((len(lambda_values), len(pitch_values_deg)), np.nan)
    for _, row in summary_df.iterrows():
        valid = row["n_nonconverged"] == 0
        if high_induction_model == "reference_glauert":
            valid = valid and row["n_physically_invalid"] == 0
        if valid and np.isfinite(row["Cp"]):
            i = int(np.argmin(np.abs(lambda_values - row["lambda"])))
            j = int(np.argmin(np.abs(pitch_values_deg - row["pitch_deg"])))
            cp_filtered[i, j] = row["Cp"]
    return cp_filtered


def save_cp_surface_outputs(cp_raw, cp_filtered, summary_df):
    """Save compact Cp surface CSVs."""
    pd.DataFrame(
        cp_raw,
        index=pd.Index(lambda_values, name="lambda"),
        columns=pd.Index(pitch_values_deg, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, "cp_surface_raw.csv"))
    pd.DataFrame(
        cp_filtered,
        index=pd.Index(lambda_values, name="lambda"),
        columns=pd.Index(pitch_values_deg, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, "cp_surface_filtered.csv"))
    summary_df.to_csv(os.path.join(output_dir, "lambda_beta_summary.csv"), index=False)


def plot_surface_and_contour(surface, prefix, title):
    """Plot 3D surface and contour for one Cp field."""
    lam_grid, pitch_grid = np.meshgrid(lambda_values, pitch_values_deg, indexing="ij")
    plot_surface = np.where(surface < cp_min_plot, np.nan, surface)
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(pitch_grid, lam_grid, plot_surface, cmap="viridis")
    ax.set_xlabel("pitch [deg]")
    ax.set_ylabel("lambda")
    ax.set_zlabel("Cp")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    save_figure(fig, f"{prefix}_plot.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    contour = ax.contourf(lam_grid, pitch_grid, plot_surface, levels=30, cmap="viridis")
    ax.set_xlabel("lambda")
    ax.set_ylabel("pitch [deg]")
    ax.set_title(title)
    fig.colorbar(contour, ax=ax, label="Cp")
    save_figure(fig, f"{prefix}_contour.png")


def interpolate_cp_surface(cp_filtered):
    """Interpolate from valid non-negative filtered Cp values only."""
    lam_grid, pitch_grid = np.meshgrid(lambda_values, pitch_values_deg, indexing="ij")
    valid = np.isfinite(cp_filtered) & (cp_filtered >= 0.0)
    cp_interpolated = cp_filtered.copy()
    cp_interpolated[cp_interpolated < 0.0] = np.nan
    mask = np.full(cp_filtered.shape, 2, dtype=int)
    mask[valid] = 0
    if valid.sum() >= 3:
        points = np.column_stack((lam_grid[valid], pitch_grid[valid]))
        values = cp_filtered[valid]
        target = np.column_stack((lam_grid.ravel(), pitch_grid.ravel()))
        interp = griddata(points, values, target, method="linear").reshape(cp_filtered.shape)
        fill = ~valid & np.isfinite(interp)
        cp_interpolated[fill] = interp[fill]
        mask[fill] = 1
    return cp_interpolated, mask


def save_interpolated_outputs(cp_interpolated, mask):
    """Save interpolation CSVs."""
    pd.DataFrame(
        cp_interpolated,
        index=pd.Index(lambda_values, name="lambda"),
        columns=pd.Index(pitch_values_deg, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, "cp_surface_interpolated.csv"))
    pd.DataFrame(
        mask,
        index=pd.Index(lambda_values, name="lambda"),
        columns=pd.Index(pitch_values_deg, name="pitch_deg"),
    ).to_csv(os.path.join(output_dir, "cp_surface_interpolation_mask.csv"))


def plot_interpolated_surface(cp_interpolated, mask):
    """Plot interpolated Cp outputs and optional zero-filled visualization."""
    plot_surface = cp_interpolated.copy()
    if interpolation_visual_zero_fill:
        visual = np.where(np.isnan(plot_surface), 0.0, plot_surface)
        visual = np.where(visual < 0.0, 0.0, visual)
        pd.DataFrame(
            visual,
            index=pd.Index(lambda_values, name="lambda"),
            columns=pd.Index(pitch_values_deg, name="pitch_deg"),
        ).to_csv(os.path.join(output_dir, "cp_surface_interpolated_visual.csv"))
        plot_surface = visual
        title = "Interpolated Cp surface - visualization only"
    else:
        plot_surface[plot_surface < 0.0] = np.nan
        title = "Interpolated Cp surface"
    plot_surface_and_contour(plot_surface, "cp_surface_interpolated", title)

    lam_grid, pitch_grid = np.meshgrid(lambda_values, pitch_values_deg, indexing="ij")
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = ListedColormap(["#1f77b4", "#ffbf00", "#d9d9d9"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    mesh = ax.pcolormesh(
        lam_grid, pitch_grid, mask,
        cmap=cmap, norm=norm, shading="nearest",
    )
    ax.set_xlabel("lambda")
    ax.set_ylabel("pitch [deg]")
    ax.set_title("Interpolation mask")
    fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2])
    save_figure(fig, "cp_surface_interpolation_mask_plot.png")


def run_pitch_sweep(solver, blade):
    """Run fixed-wind-speed Cp/Ct pitch-family sweep."""
    rows = []
    for pitch_deg in tqdm(pitch_sweep_deg, desc="Pitch family sweep"):
        for lam in lambda_sweep_values:
            omega = lam * pitch_sweep_vinf / blade.R
            result = solver.solve_rotor(
                V_inf=pitch_sweep_vinf,
                omega=omega,
                pitch=np.radians(pitch_deg),
            )
            summary = summarize_rotor_result(result)
            rows.append({
                "lambda": lam,
                "pitch_deg": pitch_deg,
                "Cp": summary["Cp"],
                "Ct": summary["Ct"],
                "torque": summary["torque"],
                "thrust": summary["thrust"],
                "n_nonconverged": summary["n_nonconverged"],
                "valid": summary["n_nonconverged"] == 0,
                "invalid_reason": (
                    "non_converged_sections" if summary["n_nonconverged"] > 0 else ""
                ),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "cp_curve_pitch_sweep.csv"), index=False)
    return df


def plot_pitch_sweep(df):
    """Plot Cp and Ct pitch families."""
    for col, ylabel, filename in [
        ("Cp", "Cp", "cp_vs_lambda_pitch_family.png"),
        ("Ct", "Ct", "ct_vs_lambda_pitch_family.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for pitch_deg, group in df.groupby("pitch_deg"):
            ax.plot(group["lambda"], group[col], "-o", ms=3, label=f"{pitch_deg:g} deg")
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs lambda by pitch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_figure(fig, filename)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    make_output_dir(output_dir)
    blade, aero, solver = load_blade_and_aero()

    plot_geometry(blade)
    plot_airfoil_polars(aero)

    print(
        f"Running Cp surface: {len(lambda_values)} x {len(pitch_values_deg)} "
        f"= {len(lambda_values) * len(pitch_values_deg)} cases"
    )
    cp_raw, summary_df = run_cp_surface(solver, blade)
    cp_filtered = build_filtered_surface(summary_df) if create_filtered_surface else cp_raw
    save_cp_surface_outputs(cp_raw, cp_filtered, summary_df)
    plot_surface_and_contour(
        cp_filtered, "cp_surface_filtered", "Filtered Cp surface"
    )

    if create_interpolated_surface:
        cp_interpolated, mask = interpolate_cp_surface(cp_filtered)
        save_interpolated_outputs(cp_interpolated, mask)
        plot_interpolated_surface(cp_interpolated, mask)

    print(
        f"Running pitch sweep: {len(pitch_sweep_deg)} x {len(lambda_sweep_values)} "
        f"= {len(pitch_sweep_deg) * len(lambda_sweep_values)} cases"
    )
    pitch_sweep_df = run_pitch_sweep(solver, blade)
    plot_pitch_sweep(pitch_sweep_df)

    print(f"Simple run complete. Outputs saved in: {output_dir}")
