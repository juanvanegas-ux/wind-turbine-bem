import numpy as np
import pandas as pd


class ViternaExtrapolator:
    """
    360 degree polar extrapolation with the Viterna-Corrigan (1982) method.

    a couple of implementation notes:

    the Viterna equations are only valid for alpha in [alpha_stall, 90 deg].
    past 90 deg the usual practice (NREL AeroDyn, OpenFAST) is:

        Cl(a) = -Cl(180 - a)          (antisymmetric about 90)
        Cd(a) =  Cd(180 - a)          (symmetric about 90)

    that way the B2*cos(a) term does not give negative Cd near 0 and +/-180,
    which happens whenever cd_stall is small (thin airfoils at low Re).

    the negative alpha side is the same equations on |alpha| with Cl sign
    flipped, so the polar stays antisymmetric in Cl and symmetric in Cd like the
    physics wants.
    """

    def __init__(self, geometry=None, cd90=None, AR=None, step_deg=1.0):
        self.geometry  = geometry
        self.cd90      = cd90
        self.AR        = AR
        self.step_deg  = step_deg

        if self.AR is None and self.geometry is not None:
            self.AR = self.calculate_aspect_ratio_from_geometry()

    # --------------------------------------------------
    # Geometry helpers
    # --------------------------------------------------

    def calculate_aspect_ratio_from_geometry(self):
        if self.geometry.r is None or self.geometry.chord is None:
            raise ValueError("Geometry must be loaded before calculating AR.")
        if self.geometry.R is None:
            raise ValueError("Geometry.R is not defined.")

        r     = self.geometry.r
        chord = self.geometry.chord
        R     = self.geometry.R

        blade_area = np.trapezoid(chord, r)
        if blade_area <= 0:
            raise ValueError("Blade area must be positive.")

        return R ** 2 / blade_area

    def estimate_cd90(self):
        if self.cd90 is not None:
            return self.cd90
        if self.AR is None:
            raise ValueError(
                "Provide cd90, AR, or a loaded BladeGeometry object."
            )
        self.cd90 = 1.11 + 0.018 * self.AR
        return self.cd90

    # --------------------------------------------------
    # Stall angle detection
    # --------------------------------------------------

    def find_stall_angles(self, polar_df):
        """
        Detect the positive and negative stall angles from the measured polar.

        the stall angle is just the angle of MAX Cl (positive side) or MIN Cl
        (negative side) in the measured data. that is correct by definition and
        does not get fooled by mid range Cl inflections (like the laminar bubble
        kink in the S series airfoils such as S822/S823 around alpha 5 to 7 deg).

        an earlier gradient based detector (first dCl/dalpha zero crossing) used
        to fire on those inflections and anchor Viterna at alpha ~6 deg instead
        of the real stall around alpha ~17 deg. that made the BEM polar drift off
        the raw data over the whole alpha = 6 to 17 deg range, about a 35% under
        estimate of Cl for the sections working in that range.

        manual override: if the polar DataFrame carries attributes
        'alpha_stall_pos' or
        'alpha_stall_neg', those values are used directly (useful for debugging
        or for polars where the automatic detection is still wrong).

        non lifting airfoils: for cylinders / non lifting sections
        (|Cl|_max < 0.05) it returns +/-15 deg as nominal stall angles, so
        Viterna gets a sensible flat plate extension without divide by zero
        warnings.
        """
        alpha = polar_df["alpha"].values
        cl    = polar_df["cl"].values

        # Guard: flat polar (cylinder / non-lifting section)
        if np.abs(cl).max() < 0.05:
            return 15.0, -15.0

        # Manual override via DataFrame attributes
        if hasattr(polar_df, "attrs"):
            if "alpha_stall_pos" in polar_df.attrs:
                alpha_stall_pos = float(polar_df.attrs["alpha_stall_pos"])
            else:
                # True positive stall = angle of maximum Cl in measured data
                pos_mask = alpha > 0
                if pos_mask.any():
                    alpha_stall_pos = float(alpha[pos_mask][np.argmax(cl[pos_mask])])
                else:
                    alpha_stall_pos = float(alpha[np.argmax(cl)])

            if "alpha_stall_neg" in polar_df.attrs:
                alpha_stall_neg = float(polar_df.attrs["alpha_stall_neg"])
            else:
                # True negative stall = angle of minimum Cl in measured data
                neg_mask = alpha < 0
                if neg_mask.any():
                    alpha_stall_neg = float(alpha[neg_mask][np.argmin(cl[neg_mask])])
                else:
                    alpha_stall_neg = float(alpha[np.argmin(cl)])
        else:
            pos_mask = alpha > 0
            alpha_stall_pos = float(
                alpha[pos_mask][np.argmax(cl[pos_mask])] if pos_mask.any()
                else alpha[np.argmax(cl)]
            )
            neg_mask = alpha < 0
            alpha_stall_neg = float(
                alpha[neg_mask][np.argmin(cl[neg_mask])] if neg_mask.any()
                else alpha[np.argmin(cl)]
            )

        return alpha_stall_pos, alpha_stall_neg

    # --------------------------------------------------
    # Viterna constants  (valid for alpha in [alpha_stall, 90 deg])
    # --------------------------------------------------

    def viterna_constants(self, alpha_stall_deg, cl_stall, cd_stall):
        """
        Compute A1, A2, B1, B2 anchored at the stall point.
        alpha_stall_deg and cl_stall must be passed as positive values.
        """
        cd90    = self.estimate_cd90()
        alpha_s = np.radians(abs(alpha_stall_deg))
        sin_s   = np.sin(alpha_s)
        cos_s   = np.cos(alpha_s)

        if abs(sin_s) < 1e-6:
            sin_s = 1e-6
        if abs(cos_s) < 1e-6:
            cos_s = 1e-6

        A1 = cd90 / 2.0
        B1 = cd90

        A2 = (cl_stall - A1 * np.sin(2.0 * alpha_s)) * sin_s / (cos_s ** 2)
        B2 = (cd_stall - B1 * sin_s ** 2) / cos_s

        return A1, A2, B1, B2

    # --------------------------------------------------
    # Viterna equations  (alpha in [alpha_stall, 90 deg])
    # --------------------------------------------------

    def _viterna_cl_cd(self, alpha_rad, A1, A2, B1, B2):
        """
        Evaluate Viterna Cl and Cd. alpha_rad must be in [alpha_stall, pi/2].
        """
        sin_a = np.sin(alpha_rad)
        cos_a = np.cos(alpha_rad)
        sin_a = max(sin_a, 1e-4)           # protect A2 term at small angles

        cl = A1 * np.sin(2.0 * alpha_rad) + A2 * (cos_a ** 2 / sin_a)
        cd = B1 * sin_a ** 2 + B2 * cos_a
        cd = max(cd, 0.0)

        return cl, cd

    # --------------------------------------------------
    # full positive side polar  (alpha_stall to 180 deg)
    # --------------------------------------------------

    def _positive_side(self, alpha_deg, a_s, cl_s, cd_s, A1, A2, B1, B2):
        """
        Return (Cl, Cd) for alpha_deg in [a_s, 180 deg] on the positive side.

        the three regions:
        [a_s,  90]           Viterna directly
        [90,  180-a_s]       mirror:  Cl(a) = -Cl(180-a),  Cd(a) = Cd(180-a)
        [180-a_s, 180]       blend Cl and Cd down to 0 (edge on flat plate)
        """
        if alpha_deg <= 90.0:
            return self._viterna_cl_cd(np.radians(alpha_deg), A1, A2, B1, B2)

        # mirror angle, ends up in [0, 90]
        alpha_m = 180.0 - alpha_deg

        if alpha_m >= a_s:
            cl_m, cd_m = self._viterna_cl_cd(np.radians(alpha_m), A1, A2, B1, B2)
        else:
            # blend from the Viterna anchor at alpha_stall to flat plate at 0 deg.
            # t=0 when mirror=alpha_stall (full Viterna), t=1 when mirror=0 (flat plate)
            t = 1.0 - (alpha_m / a_s) if a_s > 0 else 1.0
            w = 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))
        
            cl_vit, cd_vit = self._viterna_cl_cd(np.radians(a_s), A1, A2, B1, B2)
            cl_fp  = np.sin(2.0 * np.radians(alpha_m))
            cd_fp  = 2.0 * np.sin(np.radians(alpha_m)) ** 2
        
            cl_m = (1.0 - w) * cl_vit + w * cl_fp
            cd_m = (1.0 - w) * cd_vit + w * cd_fp

        # Mirror symmetry rules
        cl = -cl_m
        cd =  cd_m

        # blend Cl and Cd toward (0, 0) very close to 180 deg
        # (at exactly 180 deg the airfoil is edge on, Cl=0, Cd about 0)
        EDGE_WIDTH = 15.0
        if alpha_deg > 180.0 - EDGE_WIDTH:
            w   = (alpha_deg - (180.0 - EDGE_WIDTH)) / EDGE_WIDTH
            w   = np.clip(0.5 * (1.0 - np.cos(np.pi * w)), 0.0, 1.0)
            cl  = (1.0 - w) * cl
            cd  = (1.0 - w) * cd

        return cl, max(cd, 0.0)

    # --------------------------------------------------
    # Cosine blend helper
    # --------------------------------------------------

    @staticmethod
    def _blend_weight(alpha, a_start, a_end):
        """Returns 0 at a_start, 1 at a_end (cosine taper)."""
        if a_end == a_start:
            return 1.0
        t = (alpha - a_start) / (a_end - a_start)
        t = np.clip(t, 0.0, 1.0)
        return 0.5 * (1.0 - np.cos(np.pi * t))

    # --------------------------------------------------
    # Main extrapolation
    # --------------------------------------------------

    def extrapolate(self, polar_df):
        """
        Return a 360 deg polar DataFrame (columns: alpha, cl, cd).
        """
        polar_df   = polar_df.sort_values(by="alpha").reset_index(drop=True)
        alpha_data = polar_df["alpha"].values
        cl_data    = polar_df["cl"].values
        cd_data    = polar_df["cd"].values

        # Stall detection
        alpha_stall_pos, alpha_stall_neg = self.find_stall_angles(polar_df)

        a_s_pos = abs(alpha_stall_pos)
        a_s_neg = abs(alpha_stall_neg)

        cl_sp = float(np.interp(alpha_stall_pos, alpha_data, cl_data))
        cd_sp = float(np.interp(alpha_stall_pos, alpha_data, cd_data))
        cl_sn = float(np.interp(alpha_stall_neg, alpha_data, cl_data))
        cd_sn = float(np.interp(alpha_stall_neg, alpha_data, cd_data))

        # Viterna constants. positive side uses the positive stall values, the
        # negative side mirrors so we pass abs(cl_sn) and the same cd_sn.
        A1p, A2p, B1p, B2p = self.viterna_constants(a_s_pos,  cl_sp,  cd_sp)
        A1n, A2n, B1n, B2n = self.viterna_constants(a_s_neg, -cl_sn,  cd_sn)

        BLEND_DEG = 0.5   # cosine blend width at the stall boundary. it was 3.0
                          # but a 3 deg window starts messing up the polar at
                          # alpha_stall - 3 deg, well before the real stall.

        alpha_full = np.arange(-180.0, 180.0 + self.step_deg, self.step_deg)
        cl_full    = np.zeros_like(alpha_full)
        cd_full    = np.zeros_like(alpha_full)

        for idx, a in enumerate(alpha_full):

            # ---- Inside measured range ----
            if alpha_data.min() <= a <= alpha_data.max():
                cl_v = float(np.interp(a, alpha_data, cl_data))
                cd_v = float(np.interp(a, alpha_data, cd_data))

                # Blend toward Viterna near positive stall boundary
                if a > alpha_stall_pos - BLEND_DEG:
                    cl_e, cd_e = self._positive_side(
                        a, a_s_pos, cl_sp, cd_sp, A1p, A2p, B1p, B2p
                    )
                    w    = self._blend_weight(
                        a, alpha_stall_pos - BLEND_DEG, alpha_stall_pos
                    )
                    cl_v = (1 - w) * cl_v + w * cl_e
                    cd_v = (1 - w) * cd_v + w * cd_e

                # Blend toward Viterna near negative stall boundary
                if a < alpha_stall_neg + BLEND_DEG:
                    cl_pos, cd_pos = self._positive_side(
                        -a, a_s_neg, -cl_sn, cd_sn, A1n, A2n, B1n, B2n
                    )
                    cl_e = -cl_pos
                    cd_e =  cd_pos
                    w    = self._blend_weight(
                        a, alpha_stall_neg + BLEND_DEG, alpha_stall_neg
                    )
                    cl_v = (1 - w) * cl_v + w * cl_e
                    cd_v = (1 - w) * cd_v + w * cd_e

                cl_full[idx] = cl_v
                cd_full[idx] = max(cd_v, 0.0)

            # ---- Beyond positive stall ----
            elif a > alpha_data.max():
                cl_v, cd_v = self._positive_side(
                    a, a_s_pos, cl_sp, cd_sp, A1p, A2p, B1p, B2p
                )
                cl_full[idx] = cl_v
                cd_full[idx] = max(cd_v, 0.0)

            # ---- Beyond negative stall ----
            else:
                cl_pos, cd_pos = self._positive_side(
                    -a, a_s_neg, -cl_sn, cd_sn, A1n, A2n, B1n, B2n
                )
                cl_full[idx] = -cl_pos
                cd_full[idx] = max(cd_pos, 0.0)

        return pd.DataFrame({
            "alpha": alpha_full,
            "cl":    cl_full,
            "cd":    cd_full,
        })