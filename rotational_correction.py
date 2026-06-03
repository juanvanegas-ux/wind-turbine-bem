import numpy as np
import pandas as pd


class RotationalCorrection:
    """
    Du-Selig (1998) 3D rotational stall-delay correction.

    Modifies a 2D polar to account for centrifugal pumping and
    Coriolis effects on a rotating blade section.

    Reference:
        Du, Z. and Selig, M.S. (1998). "A 3-D stall-delay model for horizontal
        axis wind turbine performance prediction." AIAA-98-0021.

    The correction factors are (Du & Selig 1998, Eq. 20):

        fL = 1/(2pi) * [ (1.6*(c/r) / 0.1267)
                         * (a - (c/r)^(d*R/(Lambda*r)))
                         / (b + (c/r)^(d*R/(Lambda*r)))
                         - 1 ]

        fD = same factor (Du & Selig 1998)

    Standard values: a = b = d = 1.0

    Lambda = tip speed ratio = omega*R / sqrt(V_inf^2 + (omega*R)^2)

    Only applied where Cl_linear > Cl_2d (post-stall region).
    Effect scales with c/r — strongest at root, zero at tip.
    """

    def __init__(
        self,
        c_over_r,
        local_tsr,
        R_over_r,
        tip_tsr,
        alpha_0_deg=-2.0,
        Cl_slope=2 * np.pi,
        a_coeff=1.0,
        b_coeff=1.0,
        d_coeff=1.0,
    ):
        """
        Parameters
        ----------
        c_over_r    : float  local chord / local radius
        local_tsr   : float  omega * r / V_inf  at this section
        R_over_r    : float  R / r  at this section
        tip_tsr     : float  omega * R / V_inf  (global TSR)
        alpha_0_deg : float  zero-lift angle of attack [deg]
        Cl_slope    : float  lift curve slope [1/rad], default 2pi
        a_coeff     : float  Du-Selig model constant, default 1.0
        b_coeff     : float  Du-Selig model constant, default 1.0
        d_coeff     : float  Du-Selig model constant, default 1.0
        """
        self.cr        = c_over_r
        self.lsr       = max(local_tsr, 0.01)
        self.Rr        = R_over_r
        self.tsr       = max(tip_tsr, 0.01)
        self.alpha_0   = np.radians(alpha_0_deg)
        self.Cl_slope  = Cl_slope
        self.a         = a_coeff
        self.b         = b_coeff
        self.d         = d_coeff

        # Global Lambda (tip speed ratio based on resultant velocity)
        self.Lambda = self.tsr / np.sqrt(1.0 + self.tsr**2)

        self.fL, self.fD = self._compute_factors()

    def _compute_factors(self):
        """
        Compute Du-Selig (1998) correction factors fL and fD.

        Canonical Equation 20 from Du & Selig (1998), as cited in the AeroDyn
        and QBlade theory manuals:

            fL = 1/(2π) * [ (1.6*(c/r) / 0.1267)
                            * (a - (c/r)^exp) / (b + (c/r)^exp)
                            - 1 ]

        where exp = d * (R/r) / Λ and a = b = d = 1 are the standard model
        constants.

        Notes
        -----
        - The (a - cr^exp) numerator does NOT cancel: it multiplies the
          1.6(c/r)/0.1267 prefactor and is divided by (b + cr^exp) (note the
          PLUS sign), and the whole bracket carries a trailing -1.
        - max(fL, 0) keeps only the physical (stall-delay enhancing) part:
          fL naturally goes negative toward the tip (small c/r), where the
          correction should vanish.

        History: a previous implementation used den = (b - cr^exp), cancelled
        the numerator, dropped the trailing -1, and carried a stray factor of
        π. That form over-predicted fL by ~7x at c/r = 0.15. Fixed (F2) to the
        canonical Eq. 20 above.
        """
        cr = self.cr
        a  = self.a
        b  = self.b
        d  = self.d
        Rr = self.Rr    # R/r
        L  = self.Lambda

        # Exponent: d * R / (Λ * r) = d * (R/r) / Λ
        exponent = d * Rr / max(L, 1e-6)

        # cr^exponent — protect against overflow
        cr_exp = cr ** min(exponent, 50.0)

        # fL — Du-Selig (1998) Eq. 20 (canonical form)
        den_L = b + cr_exp
        if abs(den_L) < 1e-10:
            fL = 0.0
        else:
            fL = (1.0 / (2.0 * np.pi)) * (
                (1.6 * cr / 0.1267) * (a - cr_exp) / den_L - 1.0
            )

        fL = max(fL, 0.0)

        # fD uses the same factor (Du & Selig 1998)
        fD = fL

        return fL, fD

    def apply(
        self,
        polar_df,
        alpha_min_deg=-25.0,
        alpha_max_deg=25.0,
        delta_cl_max=0.3,
    ):
        """
        Apply correction to a 360° polar DataFrame.
        Returns a new DataFrame with corrected cl and cd.

        For validation use, callers should pass measured-range polar data only,
        then extrapolate afterward if a 360-degree polar is required.

        Only modifies angles where Cl_linear > Cl_2d (post-stall).
        Inside the attached flow region the correction is zero by construction.

        Parameters
        ----------
        alpha_min_deg : float
            Minimum angle of attack [deg] affected by the correction.
        alpha_max_deg : float
            Maximum angle of attack [deg] beyond which the correction is
            not applied. Default 30°. Above this angle the 2D flat-plate
            model already captures separated flow adequately and the
            3D rotational effect saturates.
        delta_cl_max  : float
            Maximum allowed delta_Cl. Prevents unbounded correction at
            very high angles of attack where Cl_linear diverges.
            Default 0.5 (approximately half of S809 Cl_max).
        """
        alpha  = polar_df["alpha"].values
        Cl     = polar_df["cl"].values.copy()
        Cd     = polar_df["cd"].values.copy()

        mask_attached = (polar_df["alpha"] >= -5.0) & (polar_df["alpha"] <= 10.0)
        if mask_attached.any():
            Cd_min = float(polar_df.loc[mask_attached, "cd"].min())
        else:
            Cd_min = float(polar_df["cd"].min())
            
        for i, a_deg in enumerate(alpha):

            # Only apply in the correction region
            if a_deg < alpha_min_deg or a_deg > alpha_max_deg:
                continue

            # Linear (potential flow) Cl prediction
            Cl_lin = self.Cl_slope * (np.radians(a_deg) - self.alpha_0)

            # Correction only where rotation helps: Cl_lin > Cl_2d
            raw_delta_Cl = self.fL * max(Cl_lin - Cl[i], 0.0)
            raw_delta_Cd = self.fD * max(Cd[i] - Cd_min, 0.0)

            # Apply caps
            delta_Cl = min(raw_delta_Cl, delta_cl_max)
            delta_Cd = min(raw_delta_Cd, delta_cl_max)

            Cl[i] += delta_Cl
            Cd[i]  = max(Cd[i] - delta_Cd, Cd_min)

        return pd.DataFrame({"alpha": alpha, "cl": Cl, "cd": Cd})
