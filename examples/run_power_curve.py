"""Design a small wind turbine, then compute and plot its BEM performance.

Outputs (saved to ../results):
  - cp_lambda.png : power coefficient vs tip-speed ratio
  - power_curve.png : electrical-power curve vs wind speed
Run:  python examples/run_power_curve.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bem import design_optimum_blade, evaluate, cp_lambda_curve  # noqa: E402

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

RHO = 1.225          # air density [kg/m^3]
RATED_POWER = 5_000  # generator rating [W]
CUT_IN, CUT_OUT = 3.0, 25.0


def main():
    rotor = design_optimum_blade(radius=4.0, hub_radius=0.4, n_blades=3,
                                 tsr_design=7.0, cl_design=1.0)
    print(f"Rotor: {rotor.n_blades} blades, R = {rotor.radius} m, "
          f"swept area = {rotor.swept_area:.1f} m^2")

    # --- Cp vs tip-speed ratio -------------------------------------------
    tsr = np.linspace(2, 12, 41)
    cp = cp_lambda_curve(rotor, tsr, wind_speed=8.0, rho=RHO)
    i_opt = int(np.argmax(cp))
    tsr_opt, cp_max = tsr[i_opt], cp[i_opt]
    print(f"Optimum TSR = {tsr_opt:.2f}, Cp,max = {cp_max:.3f} "
          f"(Betz limit = 0.593)")

    plt.figure(figsize=(7, 4.5))
    plt.plot(tsr, cp, lw=2, color="#1f4e79")
    plt.axhline(16 / 27, ls="--", color="grey", label="Betz limit (0.593)")
    plt.scatter([tsr_opt], [cp_max], color="#c0392b", zorder=5,
                label=f"Cp,max = {cp_max:.3f} @ TSR {tsr_opt:.1f}")
    plt.xlabel("Tip-speed ratio  λ")
    plt.ylabel("Power coefficient  $C_p$")
    plt.title("Rotor power coefficient (BEM)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "cp_lambda.png"), dpi=130)

    # --- Power curve (variable-speed, optimal-TSR, pitch-regulated) -------
    winds = np.arange(CUT_IN, CUT_OUT + 0.1, 0.5)
    powers = []
    for u in winds:
        omega = tsr_opt * u / rotor.radius          # track optimal TSR
        p = evaluate(rotor, u, omega, RHO)["power"]
        powers.append(min(max(p, 0.0), RATED_POWER))  # cap at rated
    powers = np.array(powers)

    rated_idx = np.argmax(powers >= RATED_POWER * 0.999)
    u_rated = winds[rated_idx] if powers.max() >= RATED_POWER * 0.999 else None
    aep_kwh = np.trapezoid(powers, winds) / 1000 * 24 * 365 / (CUT_OUT - CUT_IN)
    print(f"Rated wind speed ~ {u_rated} m/s" if u_rated else "Rated not reached")

    plt.figure(figsize=(7, 4.5))
    plt.plot(winds, powers / 1000, lw=2, color="#1f4e79")
    plt.axhline(RATED_POWER / 1000, ls="--", color="grey",
                label=f"Rated power ({RATED_POWER/1000:.0f} kW)")
    if u_rated:
        plt.axvline(u_rated, ls=":", color="#c0392b",
                    label=f"Rated wind ≈ {u_rated:.1f} m/s")
    plt.xlabel("Wind speed  [m/s]")
    plt.ylabel("Electrical power  [kW]")
    plt.title("Wind-turbine power curve (BEM)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "power_curve.png"), dpi=130)

    print(f"Saved plots to {os.path.abspath(RESULTS)}")


if __name__ == "__main__":
    main()
