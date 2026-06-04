"""
compare the smart blade against the comercial one, same solver for both.

i run both blades through the exact same BEM setup (same corrections, same air
density, same number of sections) so the only thing that changes is the geometry
and the airfoils. that way the comparison is actually fair.

what comes out (figure compare_blades.png, 2x3 panels):
  1. chord  c(r/R)
  2. twist  beta(r/R)
  3. Cp(lambda)   at beta = 0
  4. Ct(lambda)   at beta = 0
  5. power P(V)   at a common fixed rpm
  6. thrust T(V)  at a common fixed rpm

it also prints a small summary table (R, swept area, solidity, peak Cp,
lambda_opt) and dumps two csv files.

both blades use r_is_span_offset=False, so the 'r' column is the real rotor
radius (smart R=2.5 m, comercial R=2.275 m). Cp and Ct are dimensionless and
each one is normalised by its own swept area, so they compare directly. the
P(V)/T(V) panels use each rotor own area but the same shaft speed.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Geometria import BladeGeometry
from Aero import Aero
from BEM import BEMSolver

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Outputs", "Compare")
os.makedirs(OUT, exist_ok=True)

# shared physics / solver settings, same for both blades
RHO          = 1.225
B            = 3
HUB_D        = 0.2
N_SECTIONS   = 50
PITCH        = 0.0
RPM_COMMON   = 250.0                      # rotor speed used for the P(V)/T(V) panels
OMEGA_COMMON = RPM_COMMON * 2*np.pi/60.0
V_CP         = 8.0                        # wind speed for the Cp(lambda) sweep
LAMBDA_GRID  = np.linspace(1.0, 11.0, 22)
V_GRID       = np.arange(3.0, 20.5, 1.0)

SOLVER_KW = dict(rho=RHO, max_iter=500, tol=1e-3, relaxation=0.15,
                 use_tip_loss=True, use_root_loss=True, use_glauert=True,
                 solver_method="picard")

# airfoil polars for each blade
SMART_POLARS = {
    "cylinder": "Inputs/smart/cylinder.csv",
    "S822":     "Inputs/smart/S822.csv",
    "S823":     "Inputs/smart/S823.csv",
}
COM_POLARS = {
    "cylinder": "Inputs/comercial/cylinder.csv",
    "cba":      "Inputs/comercial/cba.csv",   # only +/-25 deg, so Viterna fills the rest
}

BLADES = {
    "Smart":     dict(geom="Inputs/smart/bladegeom_smart.csv",         polars=SMART_POLARS, color="tab:blue"),
    "Comercial": dict(geom="Inputs/comercial/bladegeom_comercial.csv", polars=COM_POLARS,   color="tab:red"),
}


def build(geom_file, polars):
    blade = BladeGeometry(file_path=geom_file, n_sections=N_SECTIONS, B=B,
                          hub_diameter=HUB_D, r_is_span_offset=False)
    blade.load_csv(); blade.Malla()
    aero = Aero(geometry=blade, polar_files=polars, extrapolate_360=True,
                step_deg=1.0)
    aero.load_polars()
    solver = BEMSolver(geometry=blade, aero=aero, **SOLVER_KW)
    return blade, aero, solver


def cp_lambda(solver, R):
    """Cp, Ct across the tip speed ratio grid at fixed V_CP, beta=0."""
    cp, ct = [], []
    for lam in LAMBDA_GRID:
        omega = lam * V_CP / R
        try:
            res = solver.solve_rotor(V_inf=V_CP, omega=omega, pitch=PITCH)
            cp.append(res["Cp"]); ct.append(res["Ct"])
        except Exception:
            # if a point does not converge just leave a hole in the curve
            cp.append(np.nan); ct.append(np.nan)
    return np.array(cp), np.array(ct)


def power_curve(solver, R):
    """P, T, Q across the wind grid at the common fixed rpm, beta=0."""
    P, T, Q = [], [], []
    for v in V_GRID:
        try:
            res = solver.solve_rotor(V_inf=v, omega=OMEGA_COMMON, pitch=PITCH)
            P.append(res["power"]); T.append(res["thrust"]); Q.append(res["torque"])
        except Exception:
            P.append(np.nan); T.append(np.nan); Q.append(np.nan)
    return np.array(P), np.array(T), np.array(Q)


def main():
    data = {}
    for name, cfg in BLADES.items():
        print(f"\n=== building {name} ===")
        blade, aero, solver = build(cfg["geom"], cfg["polars"])
        R = blade.R
        area = np.pi * R**2
        # geometry distributions straight from the csv (true radius)
        g = pd.read_csv(cfg["geom"]); g.columns = g.columns.str.strip()
        # planform solidity = B * mean_chord * span / area
        span = R - g["r"].min()
        mean_c = np.trapezoid(g["chord"], g["r"]) / (g["r"].max() - g["r"].min())
        solidity = B * mean_c * span / area
        print(f"  R={R:.3f} m  area={area:.3f} m^2  solidity~{solidity:.3f}")
        print(f"  Cp(lambda) sweep ({len(LAMBDA_GRID)} pts)...")
        cp, ct = cp_lambda(solver, R)
        ilam = int(np.nanargmax(cp))
        print(f"  Cp_max={cp[ilam]:.3f} at lambda={LAMBDA_GRID[ilam]:.2f}")
        print(f"  P(V) sweep at {RPM_COMMON:.0f} RPM ({len(V_GRID)} pts)...")
        P, T, Q = power_curve(solver, R)
        data[name] = dict(blade=blade, R=R, area=area, solidity=solidity,
                          geom=g, cp=cp, ct=ct, lam_opt=LAMBDA_GRID[ilam],
                          cp_max=cp[ilam], P=P, T=T, Q=Q, color=cfg["color"])

    # summary table
    print("\n" + "=" * 64)
    print(f"{'metric':<22}" + "".join(f"{n:>20}" for n in BLADES))
    print("-" * 64)
    def row(lbl, fn):
        print(f"{lbl:<22}" + "".join(f"{fn(data[n]):>20}" for n in BLADES))
    row("Rotor radius R [m]",  lambda d: f"{d['R']:.3f}")
    row("Swept area [m^2]",    lambda d: f"{d['area']:.3f}")
    row("Solidity [-]",        lambda d: f"{d['solidity']:.3f}")
    row("Peak Cp [-]",         lambda d: f"{d['cp_max']:.3f}")
    row("lambda_opt [-]",      lambda d: f"{d['lam_opt']:.2f}")
    row(f"P@{int(V_CP)}m/s,RPM{int(RPM_COMMON)} [W]",
        lambda d: f"{d['P'][int(np.argmin(np.abs(V_GRID-V_CP)))]:.0f}")
    print("=" * 64)

    # csv dumps
    pd.DataFrame({"lambda": LAMBDA_GRID,
                  **{f"Cp_{n}": data[n]["cp"] for n in BLADES},
                  **{f"Ct_{n}": data[n]["ct"] for n in BLADES}}
                 ).to_csv(os.path.join(OUT, "compare_cp_lambda.csv"), index=False)
    pd.DataFrame({"V_mps": V_GRID,
                  **{f"P_{n}_W": data[n]["P"] for n in BLADES},
                  **{f"T_{n}_N": data[n]["T"] for n in BLADES},
                  **{f"Q_{n}_Nm": data[n]["Q"] for n in BLADES}}
                 ).to_csv(os.path.join(OUT, "compare_power_curve.csv"), index=False)

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle("Smart vs Comercial blade - BEM comparison "
                 f"(B={B}, rho={RHO}, beta=0 deg)", fontsize=14)

    for n in BLADES:
        d = data[n]; c = d["color"]; g = d["geom"]; rR = g["r"] / d["R"]
        lbl = f"{n} (R={d['R']:.2f} m)"
        ax[0, 0].plot(rR, g["chord"], "-o", ms=3, color=c, label=lbl)
        ax[0, 1].plot(rR, g["twist"], "-o", ms=3, color=c, label=lbl)
        ax[0, 2].plot(LAMBDA_GRID, d["cp"], "-o", ms=4, color=c,
                      label=f"{n}: Cp_max={d['cp_max']:.3f}@λ={d['lam_opt']:.1f}")
        ax[1, 0].plot(LAMBDA_GRID, d["ct"], "-o", ms=4, color=c, label=n)
        ax[1, 1].plot(V_GRID, d["P"] / 1e3, "-o", ms=4, color=c, label=n)
        ax[1, 2].plot(V_GRID, d["T"], "-o", ms=4, color=c, label=n)

    ax[0, 0].set(title="Chord distribution", xlabel="r/R [-]", ylabel="chord [m]")
    ax[0, 1].set(title="Twist distribution", xlabel="r/R [-]", ylabel="twist [deg]")
    ax[0, 2].set(title=f"Power coefficient (V={V_CP:.0f} m/s, β=0)",
                 xlabel="tip-speed ratio λ [-]", ylabel="Cp [-]")
    ax[0, 2].axhline(16/27, color="grey", lw=0.7, ls=":", label="Betz")
    ax[1, 0].set(title="Thrust coefficient (β=0)",
                 xlabel="tip-speed ratio λ [-]", ylabel="Ct [-]")
    ax[1, 1].set(title=f"Aerodynamic power ({RPM_COMMON:.0f} RPM, β=0)",
                 xlabel="wind speed [m/s]", ylabel="power [kW]")
    ax[1, 2].set(title=f"Rotor thrust ({RPM_COMMON:.0f} RPM, β=0)",
                 xlabel="wind speed [m/s]", ylabel="thrust [N]")
    for a in ax.ravel():
        a.grid(True, alpha=0.3); a.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT, "compare_blades.png")
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")
    print(f"wrote {os.path.join(OUT, 'compare_cp_lambda.csv')}")
    print(f"wrote {os.path.join(OUT, 'compare_power_curve.csv')}")


if __name__ == "__main__":
    main()
