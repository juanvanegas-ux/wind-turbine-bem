"""
Unsteady BEM driver for time-domain aerodynamic simulations.

Wraps a quasi-steady BEMSolver and optionally adds Øye dynamic inflow.
This is the module to use when coupling the aerodynamic model to a
time-domain simulator (e.g. Modelica via FMU, or a Python ODE driver).

Usage (Python-only):
    aero_engine = UnsteadyBEM(solver, dynamic=True)
    aero_engine.initialise(V0=8.0, omega0=20.0, pitch0_deg=2.0)
    for t, V, omega, pitch_deg in trajectory:
        out = aero_engine.step(V, omega, np.radians(pitch_deg), dt)
        # out["power"], out["torque"], out["thrust"], etc.

Two modes:
    dynamic=False  → quasi-steady: BEM is solved at every call,
                     identical to MAIN_performance behaviour.
    dynamic=True   → Øye 2-pole filter on axial induction.  BEM is still
                     used to find the *quasi-steady* induction at each
                     step, then the dynamic state is integrated.
                     Forces are recomputed from the lagged induction.
"""

import numpy as np
import pandas as pd

from dynamic_inflow import DynamicInflow


class UnsteadyBEM:
    """
    Time-stepped aerodynamic engine.

    Parameters
    ----------
    solver  : BEMSolver  — pre-built quasi-steady solver
    dynamic : bool       — enable Øye dynamic inflow filter
    """

    def __init__(self, solver, dynamic=True):
        self.solver  = solver
        self.dynamic = dynamic

        self.geometry = solver.geometry
        self.aero     = solver.aero
        self.rho      = solver.rho

        self.r_arr   = self.geometry.r.copy()
        self.dr      = np.diff(np.concatenate(([self.geometry.r_root],
                                               self.geometry.r,
                                               [self.geometry.R])))[:-1]
        self.chord   = self.geometry.chord.copy()
        self.twist   = self.geometry.twist.copy()        # [rad]
        self.B       = self.geometry.B
        self.R       = self.geometry.R

        self.di = DynamicInflow(
            n_sections=len(self.r_arr),
            R=self.R,
        )

        self._t = 0.0

    # --------------------------------------------------
    # Initialisation
    # --------------------------------------------------

    def initialise(self, V0, omega0, pitch0_rad):
        """
        Snap the dynamic-inflow state to the quasi-steady solution at
        the given operating point. Avoids a startup transient.
        """
        result = self.solver.solve_rotor(V_inf=V0, omega=omega0, pitch=pitch0_rad)
        a_qs = np.array([s["a"] for s in result["sections"]])
        self.di.initialise(a_qs)
        self._t = 0.0
        return result

    # --------------------------------------------------
    # Time step
    # --------------------------------------------------

    def step(self, V_inf, omega, pitch_rad, dt):
        """
        Advance the aerodynamic state by dt and return rotor outputs.

        Returns dict with keys:
            t, V_inf, omega, pitch_rad, pitch_deg,
            power, torque, thrust, Cp, Ct,
            sections (list, one entry per BEM section, with the same
                      keys as solver.solve_rotor)
        """
        # Quasi-steady BEM solve at the current operating point
        qs = self.solver.solve_rotor(V_inf=V_inf, omega=omega, pitch=pitch_rad)
        a_qs       = np.array([s["a"]       for s in qs["sections"]])
        a_prime    = np.array([s["a_prime"] for s in qs["sections"]])

        if self.dynamic:
            # Lag the axial induction with the Øye filter.  Tangential
            # induction is kept quasi-steady (its time constant is short).
            a_dyn = self.di.step(a_qs, self.r_arr, V_inf, dt)
        else:
            a_dyn = a_qs

        # Recompute forces section-by-section using the (possibly lagged)
        # induction.  This is a one-shot evaluation, no iteration.
        out_sections = self._recompute_forces(
            a_dyn=a_dyn, a_prime=a_prime,
            V_inf=V_inf, omega=omega, pitch_rad=pitch_rad,
            qs_sections=qs["sections"],
        )

        # Integrate to global rotor outputs
        thrust = sum(s["dFn"] for s in out_sections)
        torque = sum(s["dFt"] * s["r"] for s in out_sections)
        power  = torque * omega

        rotor_area = np.pi * self.R ** 2
        denom_p    = 0.5 * self.rho * rotor_area * V_inf ** 3
        denom_t    = 0.5 * self.rho * rotor_area * V_inf ** 2

        Cp = power  / denom_p if denom_p > 0 else np.nan
        Ct = thrust / denom_t if denom_t > 0 else np.nan

        self._t += dt

        return {
            "t":         self._t,
            "V_inf":     V_inf,
            "omega":     omega,
            "rpm":       omega * 60.0 / (2.0 * np.pi),
            "pitch_rad": pitch_rad,
            "pitch_deg": np.degrees(pitch_rad),
            "power":     power,
            "torque":    torque,
            "thrust":    thrust,
            "Cp":        Cp,
            "Ct":        Ct,
            "sections":  out_sections,
        }

    # --------------------------------------------------
    # Force recomputation from lagged induction
    # --------------------------------------------------

    def _recompute_forces(self, a_dyn, a_prime, V_inf, omega, pitch_rad,
                          qs_sections):
        """
        Given dynamic induction a_dyn (and quasi-steady a'), recompute
        section forces in one pass — no fixed-point iteration.

        We re-evaluate phi, alpha, Cl, Cd from the (possibly lagged) a
        rather than reusing the converged QS values, since the lagged a
        implies a different relative-velocity triangle.
        """
        out = []
        B   = self.B
        rho = self.rho

        for i, qs in enumerate(qs_sections):
            r       = qs["r"]
            chord   = qs["chord"]
            sigma   = qs.get("sigma", B * chord / (2.0 * np.pi * r))

            a       = float(a_dyn[i])
            ap      = float(a_prime[i])

            # Velocity triangle from lagged induction
            U_axial = V_inf * (1.0 - a)
            U_tan   = omega * r * (1.0 + ap)
            W       = np.hypot(U_axial, U_tan)
            phi     = np.arctan2(U_axial, U_tan)
            twist   = self.twist[i]
            alpha   = phi - (twist + pitch_rad)

            # Look up Cl, Cd at this angle of attack
            Cl, Cd = self.aero.get_blended_coefficients(
                r=r, alpha_deg=np.degrees(alpha), section_idx=i,
            )

            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            Cn      = Cl * cos_phi + Cd * sin_phi
            Ct_loc  = Cl * sin_phi - Cd * cos_phi

            dFn = 0.5 * rho * W ** 2 * chord * Cn * B
            dFt = 0.5 * rho * W ** 2 * chord * Ct_loc * B

            out.append({
                "r":         r,
                "chord":     chord,
                "alpha":     alpha,
                "alpha_deg": np.degrees(alpha),
                "phi":       phi,
                "Cl":        Cl,
                "Cd":        Cd,
                "Cn":        Cn,
                "Ct":        Ct_loc,
                "a":         a,
                "a_prime":   ap,
                "W":         W,
                "dFn":       dFn,
                "dFt":       dFt,
                "F":         qs.get("F", 1.0),
            })

        return out

    # --------------------------------------------------
    # Convenience: full time-domain simulation in one call
    # --------------------------------------------------

    def simulate(self, t_array, V_array, omega_array, pitch_deg_array,
                 verbose=False):
        """
        Run a full simulation over t_array with prescribed inputs.
        Initialises from the first sample. Returns a DataFrame of
        rotor-level outputs (one row per time step).

        All four input arrays must be the same length.
        """
        n = len(t_array)
        assert len(V_array)      == n
        assert len(omega_array)  == n
        assert len(pitch_deg_array) == n

        # Initialise at the first sample so we don't get a startup spike
        self.initialise(
            V0=float(V_array[0]),
            omega0=float(omega_array[0]),
            pitch0_rad=np.radians(float(pitch_deg_array[0])),
        )

        rows = []
        for k in range(n):
            dt = (t_array[k] - t_array[k-1]) if k > 0 else 0.0
            if dt <= 0:
                # First step or non-monotonic time: just evaluate QS
                out = self.solver.solve_rotor(
                    V_inf=float(V_array[k]),
                    omega=float(omega_array[k]),
                    pitch=np.radians(float(pitch_deg_array[k])),
                )
                row = {
                    "t":         float(t_array[k]),
                    "V_inf":     float(V_array[k]),
                    "omega":     float(omega_array[k]),
                    "rpm":       float(omega_array[k]) * 60.0 / (2.0 * np.pi),
                    "pitch_deg": float(pitch_deg_array[k]),
                    "power":     out["power"],
                    "torque":    out["torque"],
                    "thrust":    out["thrust"],
                    "Cp":        out["Cp"],
                    "Ct":        out["Ct"],
                }
            else:
                out = self.step(
                    V_inf     = float(V_array[k]),
                    omega     = float(omega_array[k]),
                    pitch_rad = np.radians(float(pitch_deg_array[k])),
                    dt        = dt,
                )
                row = {k_: v for k_, v in out.items() if k_ != "sections"}

            rows.append(row)

            if verbose and (k % max(1, n // 20) == 0):
                print(f"  t={row['t']:6.2f}s  V={row['V_inf']:5.2f}  "
                      f"P={row['power']/1e3:6.2f} kW")

        return pd.DataFrame(rows)
