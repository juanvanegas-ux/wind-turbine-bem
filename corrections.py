# corrections.py

import numpy as np


class BEMCorrections:
    """
    Correction models used inside a Blade Element Momentum solver.

    This class contains:
    - Prandtl tip loss correction
    - Prandtl root loss correction
    - Combined Prandtl loss factor
    - Glauert / high-thrust correction
    - Induction factor relaxation
    - Numerical safety limits
    - Optional root-region damping

    The solver should call these functions during the iterative solution
    for each blade section.
    """

    def __init__(
        self,
        B,
        R,
        r_root=0.0,
        min_F=1e-4,
        a_min=-0.2,
        a_max=0.95,
        a_prime_min=-0.5,
        a_prime_max=0.5,
        relaxation_a=0.30,
        relaxation_ap=0.30,
        use_tip_loss=True,
        use_root_loss=True,
        use_glauert=True,
        use_root_cutoff=False,
        root_cutoff_fraction=0.15,
        high_induction_model="buhl",
        buhl_transition_model="smooth_blend",
        buhl_deactivation_count_required=3,
        drag_in_induction=True,
    ):
        """
        Parameters
        ----------
        B : int
            Number of blades.

        R : float
            Rotor radius [m].

        r_root : float
            Root radius or hub radius [m].

        min_F : float
            Minimum allowed Prandtl loss factor.
            Prevents division by zero near tip/root.

        a_min, a_max : float
            Numerical limits for axial induction factor.

        a_prime_min, a_prime_max : float
            Numerical limits for tangential induction factor.

        relaxation_a : float
            Relaxation weight for axial induction.
            Example: 0.30 means:
            a = 0.70*a_old + 0.30*a_new

        relaxation_ap : float
            Relaxation weight for tangential induction.

        use_tip_loss : bool
            Activate Prandtl tip loss correction.

        use_root_loss : bool
            Activate Prandtl root loss correction.

        use_glauert : bool
            Activate high-thrust correction.

        use_root_cutoff : bool
            If True, sections below root_cutoff_fraction*R can be ignored
            or damped in the solver.

        root_cutoff_fraction : float
            Fraction of rotor radius considered unreliable near root.
        """

        self.B = B
        self.R = R
        self.r_root = r_root

        self.min_F = min_F

        self.a_min = a_min
        self.a_max = a_max

        self.a_prime_min = a_prime_min
        self.a_prime_max = a_prime_max

        self.relaxation_a = relaxation_a
        self.relaxation_ap = relaxation_ap

        self.use_tip_loss = use_tip_loss
        self.use_root_loss = use_root_loss
        self.use_glauert = use_glauert

        self.use_root_cutoff = use_root_cutoff
        self.root_cutoff_fraction = root_cutoff_fraction
        self.high_induction_model = high_induction_model
        self.buhl_transition_model = buhl_transition_model
        self.buhl_deactivation_count_required = buhl_deactivation_count_required

        # whether profile drag goes into the axial induction momentum balance.
        # AeroDyn and WT_Perf leave drag out of the induction balance by default
        # (drag still shows up in the blade loads). True keeps drag in, False
        # matches the AeroDyn/WT_Perf default. either way the loads (dFn) always
        # carry drag.
        self.drag_in_induction = drag_in_induction

        if high_induction_model not in ("buhl", "reference_glauert"):
            raise ValueError(
                f"high_induction_model must be 'buhl' or 'reference_glauert', "
                f"got {high_induction_model!r}"
            )
        if buhl_transition_model not in ("hard", "latched", "smooth_blend"):
            raise ValueError(
                "buhl_transition_model must be 'hard', 'latched', or "
                f"'smooth_blend', got {buhl_transition_model!r}"
            )

    # --------------------------------------------------
    # 1. Aerodynamic force coefficients
    # --------------------------------------------------

    @staticmethod
    def normal_force_coefficient(Cl, Cd, phi):
        """
        Normal force coefficient relative to rotor plane.

        Cn = Cl*cos(phi) + Cd*sin(phi)

        This coefficient mainly contributes to thrust.
        """

        return Cl * np.cos(phi) + Cd * np.sin(phi)

    @staticmethod
    def tangential_force_coefficient(Cl, Cd, phi):
        """
        Tangential force coefficient relative to rotor plane.

        Ct = Cl*sin(phi) - Cd*cos(phi)

        This coefficient mainly contributes to torque.
        """

        return Cl * np.sin(phi) - Cd * np.cos(phi)

    # --------------------------------------------------
    # 2. Prandtl tip/root loss factors
    # --------------------------------------------------

    def tip_loss_factor(self, r, phi):
        """
        Prandtl tip loss factor.

        Corrects the infinite-blade assumption near the blade tip.

        F_tip = 2/pi * acos(exp(-f_tip))

        where:

        f_tip = B/2 * (R-r)/(r*abs(sin(phi)))
        """

        if not self.use_tip_loss:
            return 1.0

        sin_phi = np.sin(phi)

        if abs(sin_phi) < 1e-8:
            sin_phi = np.sign(sin_phi) * 1e-8 if sin_phi != 0 else 1e-8

        if r <= 0:
            return self.min_F

        f_tip = (self.B / 2.0) * (self.R - r) / (r * abs(sin_phi))

        # Prevent numerical overflow or invalid acos input
        exp_arg = np.exp(-np.clip(f_tip, 0.0, 700.0))
        exp_arg = np.clip(exp_arg, 0.0, 1.0)

        F_tip = (2.0 / np.pi) * np.arccos(exp_arg)

        return max(F_tip, self.min_F)

    def root_loss_factor(self, r, phi):
        """
        Prandtl root loss factor.

        Corrects the loss of circulation near the blade root/hub.

        F_root = 2/pi * acos(exp(-f_root))

        where:

        f_root = B/2 * (r-r_root)/(r*abs(sin(phi)))
        """

        if not self.use_root_loss:
            return 1.0

        sin_phi = np.sin(phi)

        if abs(sin_phi) < 1e-8:
            sin_phi = np.sign(sin_phi) * 1e-8 if sin_phi != 0 else 1e-8

        if r <= self.r_root:
            return self.min_F

        f_root = (self.B / 2.0) * (r - self.r_root) / (r * abs(sin_phi))

        exp_arg = np.exp(-np.clip(f_root, 0.0, 700.0))
        exp_arg = np.clip(exp_arg, 0.0, 1.0)

        F_root = (2.0 / np.pi) * np.arccos(exp_arg)

        return max(F_root, self.min_F)

    def prandtl_loss_factor(self, r, phi):
        """
        Combined Prandtl loss factor.

        F = F_tip * F_root
        """

        F_tip = self.tip_loss_factor(r, phi)
        F_root = self.root_loss_factor(r, phi)

        F = F_tip * F_root

        return max(F, self.min_F)

    # --------------------------------------------------
    # 3. Solidity and local thrust coefficient
    # --------------------------------------------------

    @staticmethod
    def local_solidity(B, chord, r):
        """
        Local blade solidity.

        sigma = B*c/(2*pi*r)
        """

        if r <= 0:
            return np.inf

        return B * chord / (2.0 * np.pi * r)

    @staticmethod
    def local_thrust_coefficient(sigma, Cn, phi):
        """
        Local thrust coefficient used by the annular streamtube.

        CT_local = sigma * Cn / sin(phi)^2

        This is useful to decide when high-thrust correction is needed.
        """

        sin_phi = np.sin(phi)

        if abs(sin_phi) < 1e-8:
            sin_phi = np.sign(sin_phi) * 1e-8 if sin_phi != 0 else 1e-8

        return sigma * Cn / (sin_phi ** 2)

    # --------------------------------------------------
    # 4. Axial induction update
    # --------------------------------------------------

    def axial_induction_momentum(self, sigma, Cn, phi, F):
        """
        Standard BEM axial induction update.

        a = 1 / [ (4 F sin^2(phi))/(sigma Cn) + 1 ]

        This is valid mainly in the low-to-moderate thrust regime.
        """

        sin_phi = np.sin(phi)

        if abs(sin_phi) < 1e-8:
            sin_phi = np.sign(sin_phi) * 1e-8 if sin_phi != 0 else 1e-8

        denominator_term = sigma * Cn

        if abs(denominator_term) < 1e-12:
            return 0.0

        a = 1.0 / (((4.0 * F * sin_phi ** 2) / denominator_term) + 1.0)

        return a

    def glauert_correction_buhl(self, CT_local, F):
        """
        High-thrust correction using a Buhl-style Glauert correction.

        This is used when the standard momentum relation becomes invalid.

        Common threshold:
        CT_local > 0.96*F

        A practical Buhl-style form:

        a = [18F - 20 - 3*sqrt(CT*(50 - 36F) + 12F*(3F - 4))]
            / [36F - 50]

        Notes:
        - This is intended for high-thrust turbulent wake states.
        - F must be positive and not too close to zero.
        - The expression inside sqrt must be protected numerically.
        """

        F = max(F, self.min_F)

        inside_sqrt = CT_local * (50.0 - 36.0 * F) + 12.0 * F * (3.0 * F - 4.0)

        # when this is actually the selected high thrust branch (CT_local >
        # 0.96*F) the thing under the sqrt is always positive. at the activation
        # threshold CT_local = 0.96*F it is 1.44*F**2 and it grows with CT_local
        # for F < 1.39. so the clip below only ever fires for evaluations whose
        # result gets thrown away (momentum branch wins) or carries a tiny
        # smooth_blend weight near CT/F = 0.93, where the clamped value is still
        # finite and continuous. it is just a safety guard, not a real invalid
        # solution, so dont treat it as a physical validity failure.
        if inside_sqrt < 0:
            inside_sqrt = 0.0

        denominator = 36.0 * F - 50.0

        if abs(denominator) < 1e-12:
            return self.a_max

        numerator = 18.0 * F - 20.0 - 3.0 * np.sqrt(inside_sqrt)

        a = numerator / denominator

        return a

    def glauert_correction_fg(self, sigma, Cn, phi, F, a_prev):
        """
        Reference-model Glauert correction via fg velocity factor.

        When a_prev > 1/3, applies:
            fg = (5 - 3*a_prev) / 4
        giving the modified momentum balance for all a:

            a * (1 - fg*a) = sigma*Cn / (4*F*sin²phi)  =: K

        Solved as the quadratic  fg*a² - a + K = 0  (fg treated as
        known from the previous iterate a_prev):

            a = [1 - sqrt(1 - 4*fg*K)] / (2*fg)

        When a_prev <= 1/3, fg = 1 and this reduces to the standard
        actuator-disk balance  a*(1-a) = K.
        """
        sin_phi = np.sin(phi)
        if abs(sin_phi) < 1e-8:
            sin_phi = 1e-8

        if abs(Cn) < 1e-12 or sigma <= 0:
            return 0.0

        K = sigma * Cn / max(4.0 * F * sin_phi ** 2, 1e-12)

        fg = 0.25 * (5.0 - 3.0 * a_prev) if a_prev > 1.0 / 3.0 else 1.0
        fg = max(fg, 0.0)

        discriminant = 1.0 - 4.0 * fg * K
        if discriminant < 0.0:
            discriminant = 0.0

        if fg < 1e-10:
            return min(K, self.a_max)

        return (1.0 - np.sqrt(discriminant)) / (2.0 * fg)

    def axial_induction(
        self,
        sigma,
        Cn,
        phi,
        F,
        glauert_active_prev=False,
        a_prev=None,
        buhl_deactivation_counter=0,
    ):

        # ``a_prev`` is the previous induction iterate (Picard). The Ning solver
        # evaluates a(phi) without an iterate and passes None; the Buhl path then
        # falls back to the momentum solution for the (1-a)^2 thrust factor (see
        # below). The reference_glauert ``fg`` scheme needs a concrete scalar, so
        # treat None as 0.0 there.
        a_prev_fg = 0.0 if a_prev is None else a_prev

        if abs(Cn) < 1e-12 or sigma <= 0:
            return 0.0, False, {
                "buhl_blend_factor": 0.0,
                "buhl_latched": False,
                "buhl_deactivation_counter": buhl_deactivation_counter,
                "CT_local_over_F": np.nan,
                "a_momentum": 0.0,
                "a_buhl": np.nan,
            }

        if self.high_induction_model == "reference_glauert":
            if not self.use_glauert:
                a_momentum = self.axial_induction_momentum(sigma, Cn, phi, F)
                return self.limit_axial_induction(a_momentum), False, {
                    "buhl_blend_factor": 0.0,
                    "buhl_latched": False,
                    "buhl_deactivation_counter": buhl_deactivation_counter,
                    "CT_local_over_F": np.nan,
                    "a_momentum": a_momentum,
                    "a_buhl": np.nan,
                }
            # Quadratic a*(1-fg*a)=K for all a.
            # fg=1 below 1/3 (standard actuator disk), fg<1 above 1/3.
            a_new = self.glauert_correction_fg(sigma, Cn, phi, F, a_prev_fg)
            glauert_active = a_prev_fg > 1.0 / 3.0
            return self.limit_axial_induction(a_new), glauert_active, {
                "buhl_blend_factor": 0.0,
                "buhl_latched": False,
                "buhl_deactivation_counter": buhl_deactivation_counter,
                "CT_local_over_F": np.nan,
                "a_momentum": np.nan,
                "a_buhl": np.nan,
            }

        # ---- Buhl path (default) ----------------------------------------
        a_momentum = self.axial_induction_momentum(sigma, Cn, phi, F)

        # careful with the axial factor here. the blade element thrust
        # coefficient that drives the high thrust (Glauert/Buhl) test and the
        # Buhl inverse is CT = sigma*(1-a)^2*Cn/sin^2(phi). the Glauert breakpoint
        # CT = 0.96*F sits at a = 0.4 only with this (1-a)^2 form. if you use the
        # bare sigma*Cn/sin^2(phi) (== CT/(1-a)^2 == 4aF/(1-a) in the momentum
        # region) the high thrust branch trips at a ~ 0.19, which drags healthy
        # outboard sections onto the Buhl branch and runs a -> a_max (no torque,
        # low Cp). so put the (1-a)^2 factor back using the current induction
        # estimate: the previous Picard iterate if we have one, else the momentum
        # solution (Ning / first call). at a_ref = a_momentum = 0.4 this gives
        # CT = 0.96*F exactly so the test stays continuous across the breakpoint.
        a_ref = a_momentum if a_prev is None else a_prev
        a_ref = min(max(float(a_ref), 0.0), 0.999)
        axial_factor = (1.0 - a_ref) ** 2
        CT_local = self.local_thrust_coefficient(sigma, Cn, phi) * axial_factor
        F_safe = max(F, self.min_F)
        CT_local_over_F = CT_local / F_safe
        a_buhl = self.glauert_correction_buhl(CT_local, F)
        buhl_blend_factor = 0.0
        buhl_latched = False
        deactivation_counter = buhl_deactivation_counter

        if not self.use_glauert:
            return self.limit_axial_induction(a_momentum), False, {
                "buhl_blend_factor": buhl_blend_factor,
                "buhl_latched": buhl_latched,
                "buhl_deactivation_counter": deactivation_counter,
                "CT_local_over_F": CT_local_over_F,
                "a_momentum": a_momentum,
                "a_buhl": a_buhl,
            }

        CT_critical_high = 0.96 * F
        CT_critical_low  = 0.93 * F

        if self.buhl_transition_model == "hard":
            if glauert_active_prev:
                use_glauert = CT_local > CT_critical_low
            else:
                use_glauert = CT_local > CT_critical_high

            if use_glauert:
                a_new = a_buhl
                buhl_blend_factor = 1.0
            else:
                a_new = a_momentum

        elif self.buhl_transition_model == "latched":
            if glauert_active_prev:
                if CT_local < CT_critical_low:
                    deactivation_counter += 1
                else:
                    deactivation_counter = 0

                use_glauert = (
                    deactivation_counter
                    < self.buhl_deactivation_count_required
                )
            else:
                use_glauert = CT_local > CT_critical_high
                deactivation_counter = 0

            if use_glauert:
                a_new = a_buhl
                buhl_blend_factor = 1.0
                buhl_latched = bool(glauert_active_prev)
            else:
                a_new = a_momentum

        elif self.buhl_transition_model == "smooth_blend":
            s = (CT_local_over_F - 0.93) / (0.96 - 0.93)
            s = np.clip(s, 0.0, 1.0)
            s = s * s * (3.0 - 2.0 * s)
            buhl_blend_factor = s
            a_new = (1.0 - s) * a_momentum + s * a_buhl
            use_glauert = s > 0.0
            deactivation_counter = 0

        else:
            raise RuntimeError(
                f"Unhandled buhl_transition_model={self.buhl_transition_model!r}"
            )

        return self.limit_axial_induction(a_new), use_glauert, {
            "buhl_transition_model": self.buhl_transition_model,
            "buhl_blend_factor": buhl_blend_factor,
            "buhl_latched": buhl_latched,
            "buhl_deactivation_counter": deactivation_counter,
            "CT_local_over_F": CT_local_over_F,
            "a_momentum": a_momentum,
            "a_buhl": a_buhl,
        }

    # --------------------------------------------------
    # 5. Tangential induction update
    # --------------------------------------------------

    def tangential_induction(self, sigma, Ct, phi, F):
        """
        Tangential induction update.

        Canonical relation:

            a' = 1 / [ (4 F sin(phi) cos(phi))/(sigma Ct) - 1 ]

        evaluated here in the mathematically identical single-denominator form

            a' = (sigma Ct) / (4 F sin(phi) cos(phi) - sigma Ct)

        Ct here is the tangential force coefficient, not the thrust coefficient.

        why i write it as one denominator: the nested form divided by sigma*Ct
        first and by (D - 1) second, and returned 0.0 at each near zero
        denominator. that second reset is a discontinuity right at the swirl pole
        4 F sinphi cosphi = sigma Ct. near the pole the true a' grows toward the
        +/- clamp, but the old code dropped it to 0 right at the crossing, so a'
        went +/-0.5 -> 0 -> -/+0.5 across iterations. that sign flip is a nice
        little source of oscillation.

        the single denominator form fixes both:
        - sigma*Ct -> 0 now gives a' -> 0 on its own (numerator -> 0), no special
          case and no inner blow up.
        - the swirl pole is handled by one sign aware floor on the single
          denominator, so a' saturates smoothly toward the clamp on the right
          side of the pole instead of snapping to 0.

        away from the pole this gives the same number as the original.
        """

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        num = sigma * Ct
        den = 4.0 * F * sin_phi * cos_phi - num

        # sign aware floor: keep |den| off zero without flipping its sign. this
        # replaces the old "-> 0 at the pole" reset and kills the +inf/-inf sign
        # flip that caused the inboard oscillation. the final magnitude is
        # bounded by limit_tangential_induction anyway (the +/-0.5 clamp).
        eps = 1e-12
        if abs(den) < eps:
            den = eps if den >= 0.0 else -eps

        a_prime = num / den

        return self.limit_tangential_induction(a_prime)

    # --------------------------------------------------
    # 6. Relaxation and limiting
    # --------------------------------------------------

    def relax_axial_induction(self, a_old, a_new):
        """
        Relax axial induction.

        a = (1-relaxation)*a_old + relaxation*a_new
        """

        a = (1.0 - self.relaxation_a) * a_old + self.relaxation_a * a_new

        return self.limit_axial_induction(a)

    def relax_tangential_induction(self, a_prime_old, a_prime_new):
        """
        Relax tangential induction.
        """

        a_prime = (
            (1.0 - self.relaxation_ap) * a_prime_old
            + self.relaxation_ap * a_prime_new
        )

        return self.limit_tangential_induction(a_prime)

    def limit_axial_induction(self, a):
        """
        Clamp axial induction to avoid nonphysical or unstable values.
        """

        if np.isnan(a) or np.isinf(a):
            return 0.0

        return np.clip(a, self.a_min, self.a_max)

    def limit_tangential_induction(self, a_prime):
        """
        Clamp tangential induction to avoid numerical explosion.

        bounds are a_prime_min / a_prime_max (default +/-0.5). they both keep us
        off the swirl pole (see tangential_induction) and cap the physical swirl.
        at very low TSR / far inboard sections the real a' can get to or past 0.5,
        so the default clamp can bias the inboard torque a bit. if you need more
        inboard swirl just widen the bounds in the constructor. i leave it at
        +/-0.5 because going wider trades inboard torque headroom for less
        protection near the pole.
        """

        if np.isnan(a_prime) or np.isinf(a_prime):
            return 0.0

        return np.clip(a_prime, self.a_prime_min, self.a_prime_max)

    # --------------------------------------------------
    # 7. Root handling
    # --------------------------------------------------

    def is_inside_root_cutoff(self, r):
        """
        Returns True if the radial station is inside the region where BEM is
        unreliable.

        The blade root usually has:
        - thick airfoils
        - cylindrical geometry
        - hub interference
        - poor aerodynamic definition
        """

        if not self.use_root_cutoff:
            return False

        return r < self.root_cutoff_fraction * self.R

    def root_weight(self, r):
        """
        Smooth weighting factor for root-region damping.

        Instead of abruptly removing the root, this can be used to gradually
        reduce aerodynamic loading near the hub.

        Returns
        -------
        weight : float
            0 near root, approaches 1 farther out.
        """

        r_cut = self.root_cutoff_fraction * self.R

        if r >= r_cut:
            return 1.0

        if r <= self.r_root:
            return 0.0

        weight = (r - self.r_root) / (r_cut - self.r_root)

        return np.clip(weight, 0.0, 1.0)

    # --------------------------------------------------
    # 8. Full correction update helper
    # --------------------------------------------------

    def update_induction_factors(
        self,
        r,
        chord,
        phi,
        Cl,
        Cd,
        a_old,
        a_prime_old,
        glauert_active_prev=False,
        buhl_deactivation_counter=0,
    ):
        Cn = self.normal_force_coefficient(Cl, Cd, phi)
        Ct = self.tangential_force_coefficient(Cl, Cd, phi)
        sigma = self.local_solidity(self.B, chord, r)
        F = self.prandtl_loss_factor(r, phi)

        # this is the value that drives the axial induction balance. with
        # drag_in_induction False we drop drag from the induction Cn
        # (Cn_ind = Cl*cos(phi)) to match the AeroDyn/WT_Perf convention. the
        # reported Cn and the blade loads (dFn, over in BEM.py) keep full drag.
        if self.drag_in_induction:
            Cn_ind = Cn
        else:
            Cn_ind = Cl * np.cos(phi)

        CT_local = self.local_thrust_coefficient(sigma, Cn_ind, phi)
        CT_critical = 0.96 * F

        a_raw, glauert_active, axial_info = self.axial_induction(
            sigma, Cn_ind, phi, F,
            glauert_active_prev=glauert_active_prev,
            a_prev=a_old,
            buhl_deactivation_counter=buhl_deactivation_counter,
        )
        a_prime_raw = self.tangential_induction(sigma, Ct, phi, F)
    
        a = self.relax_axial_induction(a_old, a_raw)
        a_prime = self.relax_tangential_induction(a_prime_old, a_prime_raw)
    
        if self.use_root_cutoff:
            weight = self.root_weight(r)
            a *= weight
            a_prime *= weight
        else:
            weight = 1.0
    
        K_debug  = CT_local / max(4.0 * F, 1e-12)
        fg_debug = 0.25 * (5.0 - 3.0 * a_old) if a_old > 1.0 / 3.0 else 1.0
        fg_debug = max(fg_debug, 0.0)
        raw_discriminant_debug = 1.0 - 4.0 * fg_debug * K_debug
        clipped_discriminant_debug = max(raw_discriminant_debug, 0.0)
        discriminant_clipped = (
            self.high_induction_model == "reference_glauert"
            and raw_discriminant_debug < 0.0
        )

        info = {
            "Cn": Cn,
            "Ct": Ct,
            "sigma": sigma,
            "F": F,
            "CT_local": CT_local,
            "CT_local_over_F": axial_info.get("CT_local_over_F", np.nan),
            "CT_critical": CT_critical,
            "glauert_active": glauert_active,
            "high_induction_active": glauert_active,
            "buhl_transition_model": axial_info.get(
                "buhl_transition_model", self.buhl_transition_model
            ),
            "buhl_blend_factor": axial_info.get("buhl_blend_factor", 0.0),
            "buhl_latched": axial_info.get("buhl_latched", False),
            "buhl_deactivation_counter": axial_info.get(
                "buhl_deactivation_counter", buhl_deactivation_counter
            ),
            "a_momentum": axial_info.get("a_momentum", np.nan),
            "a_buhl": axial_info.get("a_buhl", np.nan),
            "a_raw": a_raw,
            "a_prime_raw": a_prime_raw,
            "root_weight": weight,
            "K": K_debug,
            "fg": fg_debug,
            "raw_discriminant": raw_discriminant_debug,
            "clipped_discriminant": clipped_discriminant_debug,
            "discriminant": raw_discriminant_debug,
            "discriminant_clipped": discriminant_clipped,
            "a_old": a_old,
            "a_final": a,
            "high_induction_model": self.high_induction_model,
        }
        return a, a_prime, info

