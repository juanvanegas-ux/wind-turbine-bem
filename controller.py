import numpy as np
import pandas as pd


class WindTurbineController:
    """
    Variable speed, pitch to feather wind turbine controller.

    three operating regions:

    Region 1, MPPT  (V_cut_in <= V <= V_transition):
        track the optimal tip speed ratio lambda_design at fixed pitch
        beta_design. rotor speed climbs with the wind to keep lambda constant,
        which gives max Cp.

    Region 2, rated speed  (V_transition < V <= V_rated):
        rpm is capped at rpm_max, pitch stays at beta_design. power keeps rising
        (Cp drops a little as lambda falls below optimum).

    Region 3, rated power  (V > V_rated):
        rpm fixed at rpm_max, pitch feathers to hold P = P_rated.

    V_transition is worked out automatically as the wind speed where MPPT
    tracking would first go past rpm_max.

    Parameters
    ----------
    solver        : BEMSolver
    lambda_design : float   design tip-speed ratio
    pitch_design  : float   design pitch angle [deg]
    rotor_radius  : float   rotor radius R [m]
    v_rated       : float   rated wind speed [m/s]
    v_cut_in      : float   cut-in wind speed [m/s]
    v_cut_out     : float   cut-out wind speed [m/s]
    rpm_max       : float   maximum rotor speed [rpm]  (default: unlimited)
    beta_max_deg  : float   maximum feather pitch [deg]  (default: 35)
    feather_tol   : float   bisection tolerance as fraction of P_rated
    """

    def __init__(
        self,
        solver,
        lambda_design,
        pitch_design,
        rotor_radius,
        v_rated,
        v_cut_in,
        v_cut_out,
        rpm_max       = None,
        beta_max_deg  = 35.0,
        feather_tol   = 0.005,
        pitch_control = True,
    ):
        self.solver        = solver
        self.lambda_design = lambda_design
        self.pitch_design  = pitch_design
        self.R             = rotor_radius
        self.v_rated       = v_rated
        self.v_cut_in      = v_cut_in
        self.v_cut_out     = v_cut_out
        self.beta_max_deg  = beta_max_deg
        self.feather_tol   = feather_tol
        self.pitch_control = pitch_control   # False = passive-stall regulation

        # rpm cap. if None there is no cap (free MPPT all the way to rated)
        if rpm_max is not None:
            self.omega_max = rpm_max * 2.0 * np.pi / 60.0
            self.rpm_max   = rpm_max
        else:
            # No cap: set omega_max high enough that it never binds
            self.omega_max = lambda_design * v_rated / rotor_radius
            self.rpm_max   = self.omega_max * 60.0 / (2.0 * np.pi)

        # Rated omega is always the cap (or the MPPT value if no cap)
        self.omega_rated = self.omega_max
        self.rpm_rated   = self.rpm_max

        # Wind speed at which MPPT tracking first hits the RPM cap
        # V_transition = omega_max * R / lambda_design
        self.v_transition = self.omega_rated * rotor_radius / lambda_design

        # Compute rated power at (V_rated, omega_rated, pitch_design)
        res = self.solver.solve_rotor(
            V_inf=v_rated,
            omega=self.omega_rated,
            pitch=np.radians(pitch_design),
        )
        self.p_rated = res["power"]
        self.p_rated_non_converged = sum(
            1 for s in res["sections"] if not s.get("converged", False)
        )

    # --------------------------------------------------
    # Operating-point lookup
    # --------------------------------------------------

    def operating_point(self, v):
        """
        Return (omega [rad/s], pitch [deg]) for a given wind speed.
        Returns None for wind speeds outside the operating envelope.

        Region 1 (V <= V_transition): MPPT, lambda = lambda_design
        Region 2 (V_transition < V <= V_rated): rpm capped, pitch fixed
        Region 3 (V > V_rated):
            - if pitch_control=True:  RPM capped, pitch feathers to P_rated
            - if pitch_control=False: RPM capped, pitch stays at design
                                      (passive stall regulation)
        """
        if v < self.v_cut_in or v > self.v_cut_out:
            return None

        if v <= self.v_transition:
            # Region 1: MPPT, track optimal lambda
            omega     = self.lambda_design * v / self.R
            pitch_deg = self.pitch_design

        elif v <= self.v_rated:
            # Region 2: RPM capped, pitch fixed, power still rising
            omega     = self.omega_rated
            pitch_deg = self.pitch_design

        else:
            # Region 3
            omega = self.omega_rated
            if self.pitch_control:
                pitch_deg = self._feather_pitch(v)
            else:
                # Passive stall: pitch stays at design, power rolls off
                # naturally as the blade stalls
                pitch_deg = self.pitch_design

        return omega, pitch_deg

    def _feather_pitch(self, v):
        """
        bisection on the pitch beta in [beta_design, beta_max] so that
        P(v, omega_rated, beta) is about P_rated.
        """
        omega = self.omega_rated
        tol   = self.feather_tol * self.p_rated

        p_lo = self.solver.solve_rotor(
            V_inf=v, omega=omega,
            pitch=np.radians(self.pitch_design),
        )["power"]

        p_hi = self.solver.solve_rotor(
            V_inf=v, omega=omega,
            pitch=np.radians(self.beta_max_deg),
        )["power"]

        # Bracket checks
        if p_lo <= self.p_rated:
            return self.pitch_design   # no pitching needed
        if p_hi >= self.p_rated:
            return self.beta_max_deg   # saturated, feather fully

        lo, hi = self.pitch_design, self.beta_max_deg
        for _ in range(40):
            mid   = 0.5 * (lo + hi)
            p_mid = self.solver.solve_rotor(
                V_inf=v, omega=omega, pitch=np.radians(mid),
            )["power"]

            if abs(p_mid - self.p_rated) < tol:
                return mid
            if p_mid > self.p_rated:
                lo = mid
            else:
                hi = mid

        return 0.5 * (lo + hi)

    # --------------------------------------------------
    # Power curve sweep
    # --------------------------------------------------

    def compute_power_curve(self, wind_speeds, convergence_check_fn=None):
        """
        Evaluate the power curve over the given wind speed array.

        Parameters
        ----------
        wind_speeds          : array-like   wind speeds [m/s]
        convergence_check_fn : callable     optional, takes a BEM result
                                             dict, returns number of
                                             non-converged sections

        Returns
        -------
        pd.DataFrame with columns:
            v_inf, omega_rad_s, rpm, pitch_deg, lambda,
            Cp, Ct, power_W, torque_Nm, thrust_N,
            [non_converged if fn provided]
        """
        rows = []
        for v in wind_speeds:
            op = self.operating_point(v)
            if op is None:
                continue

            omega, pitch_deg = op
            result = self.solver.solve_rotor(
                V_inf=v,
                omega=omega,
                pitch=np.radians(pitch_deg),
            )

            row = {
                "v_inf":      v,
                "omega_rad_s": omega,
                "rpm":         omega * 60.0 / (2.0 * np.pi),
                "pitch_deg":  pitch_deg,
                "lambda":     omega * self.R / v,
                "Cp":         result["Cp"],
                "Ct":         result["Ct"],
                "power_W":    result["power"],
                "torque_Nm":  result["torque"],
                "thrust_N":   result["thrust"],
            }
            if convergence_check_fn is not None:
                row["non_converged"] = convergence_check_fn(result)

            rows.append(row)

        return pd.DataFrame(rows)

    # --------------------------------------------------
    # AEP estimate
    # --------------------------------------------------

    def compute_aep(self, power_df, weibull_c, weibull_k):
        """
        Annual energy production estimate (kWh/year) from a power curve
        DataFrame (must have 'v_inf' and 'power_W' columns) and a Weibull
        wind speed distribution.

        trapezoidal integration of:
            AEP = 8760 * integral of P(V) * f(V) dV

        where f(V) is the Weibull PDF.
        """
        v   = power_df["v_inf"].values
        p   = np.maximum(power_df["power_W"].values, 0.0)
        pdf = (weibull_k / weibull_c) * (v / weibull_c) ** (weibull_k - 1) \
              * np.exp(-(v / weibull_c) ** weibull_k)

        aep_wh = 8760.0 * np.trapezoid(p * pdf, v)
        return aep_wh / 1e3   # kWh/year

    # --------------------------------------------------
    # String representation
    # --------------------------------------------------

    def __repr__(self):
        mode = "variable-pitch" if self.pitch_control else "fixed-pitch (stall)"
        return (
            f"WindTurbineController("
            f"{mode}, "
            f"lambda={self.lambda_design:.2f}, "
            f"beta={self.pitch_design:.1f}deg, "
            f"RPM_max={self.rpm_max:.1f}, "
            f"V_transition={self.v_transition:.1f} m/s, "
            f"V_rated={self.v_rated} m/s, "
            f"P_rated={self.p_rated:.0f} W)"
        )
