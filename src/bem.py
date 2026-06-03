"""Blade Element Momentum (BEM) solver for a horizontal-axis wind turbine.

This is an original, self-contained implementation of the classic BEM method
with Prandtl tip-loss and Glauert high-induction corrections. It is meant as a
clear, readable reference rather than a production aeroelastic tool.

Author: Juan Pablo Vanegas Alzate
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Airfoil:
    """Simple analytical airfoil polar (illustrative).

    Lift is linear in angle of attack up to stall, then capped; drag follows a
    parabolic drag polar. For quantitative work, replace this with measured or
    XFOIL-generated polars loaded from a table.
    """

    cl_alpha: float = 2.0 * np.pi      # lift-curve slope [1/rad] (thin-airfoil)
    alpha_0: float = np.radians(-3.0)  # zero-lift angle of attack [rad]
    cl_max: float = 1.3                # stall lift coefficient
    cd_0: float = 0.012                # minimum drag coefficient
    k: float = 0.025                   # induced-drag factor

    def coefficients(self, alpha: np.ndarray):
        """Return (cl, cd) for angle of attack `alpha` in radians."""
        cl_linear = self.cl_alpha * (alpha - self.alpha_0)
        cl = np.clip(cl_linear, -self.cl_max, self.cl_max)
        cd = self.cd_0 + self.k * cl**2
        return cl, cd


@dataclass
class Rotor:
    """Rotor geometry and the spanwise blade definition.

    `chord` and `twist` are arrays evaluated at the radial stations `r`.
    """

    r: np.ndarray          # radial stations [m]
    chord: np.ndarray      # chord length at each station [m]
    twist: np.ndarray      # twist (incl. pitch) at each station [rad]
    n_blades: int          # number of blades
    radius: float          # tip radius [m]
    hub_radius: float      # hub radius [m]
    airfoil: Airfoil

    @property
    def swept_area(self) -> float:
        return np.pi * self.radius**2


def design_optimum_blade(
    radius: float = 4.0,
    hub_radius: float = 0.4,
    n_blades: int = 3,
    n_stations: int = 30,
    tsr_design: float = 7.0,
    cl_design: float = 1.0,
    alpha_design_deg: float = 6.0,
    airfoil: Airfoil | None = None,
) -> Rotor:
    """Generate a Betz-optimum blade (simplified Manwell formulation).

    For each station the optimum inflow angle and chord come from momentum
    theory neglecting wake rotation losses, giving a realistic tapered/twisted
    blade to feed the BEM solver.
    """
    airfoil = airfoil or Airfoil()
    r = np.linspace(hub_radius, radius, n_stations + 1)
    r = 0.5 * (r[:-1] + r[1:])  # use station midpoints

    lambda_r = tsr_design * r / radius
    phi = (2.0 / 3.0) * np.arctan(1.0 / lambda_r)              # optimum inflow
    chord = (8.0 * np.pi * r * (1.0 - np.cos(phi))) / (n_blades * cl_design)
    twist = phi - np.radians(alpha_design_deg)

    return Rotor(
        r=r,
        chord=chord,
        twist=twist,
        n_blades=n_blades,
        radius=radius,
        hub_radius=hub_radius,
        airfoil=airfoil,
    )


def _induction_factors(a, phi, sigma, cn, ct, tip_loss):
    """Axial (a) and tangential (a') induction.

    Uses the standard momentum relation in the windmill state and the Buhl
    empirical correction in the turbulent-wake state (high thrust), which keeps
    the solution well-behaved as the local thrust coefficient approaches 1.
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    ct_local = sigma * (1.0 - a) ** 2 * cn / sin_phi**2  # local thrust coeff
    if ct_local <= 0.96 * tip_loss:
        a_new = 1.0 / (1.0 + 4.0 * tip_loss * sin_phi**2 / (sigma * cn))
    else:
        a_new = (1.0 / tip_loss) * (
            0.143 + np.sqrt(max(0.0203 - 0.6427 * (0.889 - ct_local), 0.0))
        )

    denom = 4.0 * tip_loss * sin_phi * cos_phi / (sigma * ct) - 1.0
    ap_new = 1.0 / denom if abs(denom) > 1e-8 else 0.0
    return a_new, ap_new


def solve_annulus(rotor: Rotor, i: int, wind_speed, omega, rho,
                  max_iter=200, tol=1e-6, relax=0.4):
    """Solve the BEM equations at radial station `i`. Returns local loads."""
    r = rotor.r[i]
    chord = rotor.chord[i]
    twist = rotor.twist[i]
    B = rotor.n_blades
    R = rotor.radius
    sigma = B * chord / (2.0 * np.pi * r)  # local solidity

    a, ap = 0.0, 0.0
    for _ in range(max_iter):
        phi = np.arctan2((1.0 - a) * wind_speed, (1.0 + ap) * omega * r)
        if phi <= 1e-6:
            break

        # Prandtl tip-loss factor.
        f = B / 2.0 * (R - r) / (r * np.sin(phi))
        tip_loss = (2.0 / np.pi) * np.arccos(np.exp(-np.clip(f, 0, 50)))
        tip_loss = max(tip_loss, 1e-4)

        alpha = phi - twist
        cl, cd = rotor.airfoil.coefficients(alpha)
        cn = cl * np.cos(phi) + cd * np.sin(phi)   # normal (thrust) coeff
        ct = cl * np.sin(phi) - cd * np.cos(phi)   # tangential (torque) coeff

        a_new, ap_new = _induction_factors(a, phi, sigma, cn, ct, tip_loss)

        if abs(a_new - a) < tol and abs(ap_new - ap) < tol:
            a, ap = a_new, ap_new
            break
        a = (1 - relax) * a + relax * a_new
        ap = (1 - relax) * ap + relax * ap_new

    w_sq = ((1.0 - a) * wind_speed) ** 2 + ((1.0 + ap) * omega * r) ** 2
    dT = 0.5 * rho * w_sq * B * chord * cn
    dQ = 0.5 * rho * w_sq * B * chord * ct * r
    return dT, dQ, a, ap


def evaluate(rotor: Rotor, wind_speed: float, omega: float, rho: float = 1.225):
    """Integrate BEM over the blade. Returns a dict of performance values."""
    dr = np.gradient(rotor.r)
    dT = np.zeros_like(rotor.r)
    dQ = np.zeros_like(rotor.r)
    for i in range(len(rotor.r)):
        dT[i], dQ[i], _, _ = solve_annulus(rotor, i, wind_speed, omega, rho)

    thrust = np.trapezoid(dT, rotor.r)
    torque = np.trapezoid(dQ, rotor.r)
    power = omega * torque

    dyn = 0.5 * rho * rotor.swept_area * wind_speed**2
    cp = power / (dyn * wind_speed) if wind_speed > 0 else 0.0
    ct = thrust / dyn if wind_speed > 0 else 0.0
    tsr = omega * rotor.radius / wind_speed if wind_speed > 0 else 0.0

    return {
        "wind_speed": wind_speed,
        "omega": omega,
        "tsr": tsr,
        "power": power,
        "torque": torque,
        "thrust": thrust,
        "cp": cp,
        "ct": ct,
    }


def cp_lambda_curve(rotor: Rotor, tsr_values, wind_speed=8.0, rho=1.225):
    """Compute the power coefficient as a function of tip-speed ratio."""
    cps = []
    for tsr in tsr_values:
        omega = tsr * wind_speed / rotor.radius
        cps.append(evaluate(rotor, wind_speed, omega, rho)["cp"])
    return np.array(cps)
