"""
Dynamic inflow model for unsteady BEM.

Implements the Øye (1990) two-pole filter that lags the axial induction
behind the quasi-steady BEM result. Without dynamic inflow, BEM responds
instantly to changes in wind / pitch / RPM, which is unphysical and
causes power-controller chatter.

The induction state evolves as the sum of two first-order systems:

    da_int/dt = (a_qs           - a_int) / tau_1
    da_dyn/dt = (a_int + k*tau_1*da_int/dt - a_dyn) / tau_2

The slow time-constant tau_1 represents convection of the wake; the fast
tau_2 represents near-wake adjustment. Time scales follow Snel & Schepers:

    tau_1 = 1.1 / (1 - 1.3*min(a, 0.5)) * R / V_inf
    tau_2 = (0.39 - 0.26*(r/R)^2) * tau_1
    k     = 0.6

Reference:
    S. Øye, "Unsteady wake effects caused by pitch angle changes",
    IEA R&D WECS, Joint Action on Aerodynamics of Wind Turbines, 1990.
"""

import numpy as np


class DynamicInflow:
    """
    Per-section dynamic inflow filter.

    Maintains state vectors a_int[i] and a_dyn[i] for each radial section.
    Tangential induction a' is treated quasi-steady (its time constant
    is much shorter than the axial wake time scale).
    """

    def __init__(self, n_sections, R, k_coupling=0.6):
        self.n_sections = n_sections
        self.R          = R
        self.k          = k_coupling

        self.a_int = np.zeros(n_sections)   # intermediate state
        self.a_dyn = np.zeros(n_sections)   # output (filtered induction)

        self._initialised = False

    def reset(self, a_init=None):
        """Reset states. If a_init is None, set to zero."""
        if a_init is None:
            self.a_int[:] = 0.0
            self.a_dyn[:] = 0.0
        else:
            self.a_int[:] = a_init
            self.a_dyn[:] = a_init
        self._initialised = True

    def initialise(self, a_qs):
        """
        Snap states to a quasi-steady solution. Use this at t=0 to avoid
        a startup transient when beginning a simulation already in
        operation.
        """
        self.a_int[:] = a_qs
        self.a_dyn[:] = a_qs
        self._initialised = True

    def time_constants(self, a_dyn, r_arr, V_inf):
        """
        Return tau_1, tau_2 arrays (one entry per section).

        Notes
        -----
        a is clipped at 0.5 in the tau_1 formula because the original
        derivation assumes the actuator-disc relation, which breaks down
        beyond the Glauert region.  V_inf is floored at 0.1 m/s to avoid
        division by zero during cut-in transients.
        """
        a_eff = np.clip(a_dyn, 0.0, 0.5)
        V_eff = max(float(V_inf), 0.1)

        tau_1 = 1.1 / (1.0 - 1.3 * a_eff) * self.R / V_eff
        tau_2 = (0.39 - 0.26 * (r_arr / self.R) ** 2) * tau_1
        return tau_1, tau_2

    def step(self, a_qs, r_arr, V_inf, dt):
        """
        Advance the dynamic inflow states by one time step using
        explicit Euler integration.

        Parameters
        ----------
        a_qs   : array(n_sections)  quasi-steady BEM induction at t
        r_arr  : array(n_sections)  radial coordinate of each section [m]
        V_inf  : float              free-stream wind speed at t [m/s]
        dt     : float              time step [s]

        Returns
        -------
        a_dyn  : array(n_sections)  filtered induction after dt
        """
        if not self._initialised:
            self.initialise(a_qs)

        tau_1, tau_2 = self.time_constants(self.a_dyn, r_arr, V_inf)

        # Explicit Euler step on a_int
        da_int_dt   = (a_qs - self.a_int) / tau_1
        a_int_new   = self.a_int + dt * da_int_dt

        # Driver for the second filter includes the rate-of-change term
        driver      = a_int_new + self.k * tau_1 * da_int_dt
        da_dyn_dt   = (driver - self.a_dyn) / tau_2
        self.a_dyn  = self.a_dyn + dt * da_dyn_dt
        self.a_int  = a_int_new

        return self.a_dyn.copy()

    @property
    def state(self):
        """Return current (a_int, a_dyn) for logging or restart."""
        return self.a_int.copy(), self.a_dyn.copy()
