import numpy as np
import pandas as pd

from Viterna import ViternaExtrapolator
from rotational_correction import RotationalCorrection


class Aero:
    def __init__(
        self,
        geometry,
        polar_files,
        extrapolate_360=True,
        step_deg=1.0,
        apply_rotational_correction=False,
        V_ref=None,
        omega_ref=None,
        blend_half_width_frac=0.05,
        kinematic_viscosity=1.5e-5,
    ):
        """
        Parameters
        ----------
        geometry                     : BladeGeometry
        polar_files                  : dict  airfoil_name -> polar source.
                                             the value is either
                                               * a single file path (str), the
                                                 Reynolds agnostic case (this is
                                                 the usual one), OR
                                               * a dict {Re: file_path} with
                                                 several polars for the same
                                                 airfoil at different Reynolds
                                                 numbers. with more than one Re
                                                 the lookup interpolates on the
                                                 local section Reynolds number.
        extrapolate_360              : bool  run Viterna extrapolation
        step_deg                     : float polar resolution [deg]
        apply_rotational_correction  : bool  apply Du-Selig per-section correction
        V_ref                        : float reference wind speed for Du-Selig [m/s]
        omega_ref                    : float reference rotor speed for Du-Selig [rad/s]
        blend_half_width_frac        : float airfoil-transition blending zone
                                             half-width, as a fraction of R
                                             (default 0.05 = +-5% of R)
        kinematic_viscosity          : float air kinematic viscosity nu [m^2/s]
                                             used to form the local section
                                             Reynolds number Re = W c / nu when
                                             Re dependent polars are given. not
                                             used for single Re inputs.
        """
        self.geometry                    = geometry
        self.polar_files                 = polar_files
        self.extrapolate_360             = extrapolate_360
        self.step_deg                    = step_deg
        self.apply_rotational_correction = apply_rotational_correction
        self.V_ref                       = V_ref
        self.omega_ref                   = omega_ref
        self.blend_half_width_frac       = blend_half_width_frac
        self.kinematic_viscosity         = kinematic_viscosity

        self.polars = {}                  # active uncorrected 360 polars used by BEM
        self.polars_raw = {}              # original measured 2D data
        self.polars_uncorrected_360 = {}  # uncorrected BEM-ready 360 polars
        self.polars_corrected_reference = {}  # per-airfoil comparison polars
        self.polars_corrected = {}       # per-section corrected polars
        self.section_polar_map = {}
        # Re indexed 360 polars. polars_by_re[name] = list of (Re, df) sorted by
        # ascending Re. for single Re airfoils there is just one entry and the
        # lookup ignores Reynolds completely.
        self.polars_by_re = {}
        self.rotational_correction_diagnostics = []
        self.rotational_correction_applied = False
        self.rotational_correction_call_count = 0

        self.viterna = ViternaExtrapolator(
            geometry=self.geometry,
            step_deg=self.step_deg
        )

    @property
    def reynolds_dependent(self):
        """True if any airfoil has more than one Reynolds number polar.

        when it is False (the single Re default) callers should just pass
        Re=None to the lookups.
        """
        return any(len(v) > 1 for v in self.polars_by_re.values())

    def local_reynolds(self, W, chord):
        """Local section Reynolds number Re = W * chord / nu."""
        nu = self.kinematic_viscosity
        if nu is None or nu <= 0.0:
            return None
        return float(W) * float(chord) / float(nu)

    @staticmethod
    def _needs_extrapolation(df):
        """True if polar does not already cover +-170 deg or more."""
        return df["alpha"].min() > -170 or df["alpha"].max() < 170

    def _load_one_polar(self, file_path):
        """Read a single polar CSV and return the BEM-ready (raw, 360) pair."""
        df = pd.read_csv(file_path)
        df = df.rename(columns={"Cl": "cl", "Cd": "cd"})
        raw = df.copy()
        # Skip Viterna for polars already covering full 360 deg (e.g. cylinder)
        if self.extrapolate_360 and self._needs_extrapolation(df):
            df = self.viterna.extrapolate(df)
        return raw, df

    def load_polars(self):
        for airfoil_name, source in self.polar_files.items():
            if isinstance(source, dict):
                # Re keyed polars {Re: path}. sort them by Re and use the lowest
                # Re polar as the representative single Re one for the old
                # Re agnostic attributes.
                re_items = sorted(source.items(), key=lambda kv: float(kv[0]))
                by_re = []
                for re_val, path in re_items:
                    raw, df360 = self._load_one_polar(path)
                    by_re.append((float(re_val), df360.copy()))
                    if airfoil_name not in self.polars_raw:
                        # representative = lowest Re (first after sort)
                        self.polars_raw[airfoil_name] = raw.copy()
                        self.polars_uncorrected_360[airfoil_name] = df360.copy()
                        self.polars[airfoil_name] = df360.copy()
                self.polars_by_re[airfoil_name] = by_re
            else:
                # Single Re-agnostic polar (current default behaviour).
                raw, df360 = self._load_one_polar(source)
                self.polars_raw[airfoil_name] = raw.copy()
                self.polars_uncorrected_360[airfoil_name] = df360.copy()
                self.polars[airfoil_name] = df360.copy()
                # One-entry Re table (Re=None) so the lookup path is uniform.
                self.polars_by_re[airfoil_name] = [(None, df360.copy())]

        if self.apply_rotational_correction:
            self._build_corrected_polars()

    def correct_polars(
        self,
        tsr_ref,
        v_ref,
        non_lifting_threshold=0.1,
        min_r_over_R=0.0,
        max_r_over_R=1.0,
        alpha_min_deg=-25.0,
        alpha_max_deg=25.0,
        delta_cl_max=0.3,
        apply_before_viterna=False,
    ):
        """
        Build Du-Selig corrected polar artifacts without mutating baselines.

        The solver uses section-local corrected polars through
        ``section_polar_map`` when rotational correction is enabled. A
        per-airfoil corrected reference polar is also stored for diagnostics
        and plotting only.

        Parameters
        ----------
        tsr_ref               : float  representative tip-speed ratio used
                                        to set the reference omega
                                        (e.g. np.median(lambda_values))
        v_ref                 : float  reference wind speed [m/s]
                                        (should match the surface sweep speed)
        non_lifting_threshold : float  polars with max|Cl| below this value
                                        are skipped (cylinder guard)
        min_r_over_R          : float  minimum normalized radius included in
                                        airfoil-level correction averaging
        max_r_over_R          : float  maximum normalized radius included in
                                        airfoil-level correction averaging
        alpha_min_deg         : float  minimum angle of attack affected
        alpha_max_deg         : float  maximum angle of attack affected
        delta_cl_max          : float  cap on positive delta_Cl
        apply_before_viterna  : bool   if True, apply correction only to the
                                        measured polar before Viterna.
        """
        self.rotational_correction_call_count += 1
        correction_requested_more_than_once = self.rotational_correction_applied
        self.polars_corrected_reference = {}
        self.polars_corrected = {}
        self.section_polar_map = {}
        omega_ref = tsr_ref * v_ref / self.geometry.R
        diagnostics = []

        for airfoil_name, measured_polar in self.polars_raw.items():
            uncorrected_360 = self.polars_uncorrected_360[airfoil_name].copy()
            correction_source = (
                measured_polar.copy() if apply_before_viterna
                else uncorrected_360.copy()
            )
            polar_before = correction_source.copy()

            # Skip non-lifting airfoils (cylinder, blunt hub sections)
            if measured_polar["cl"].abs().max() < non_lifting_threshold:
                diagnostics.append({
                    "airfoil": airfoil_name,
                    "applied": False,
                    "reason": "non_lifting_airfoil",
                    "r_over_R_min_affected": np.nan,
                    "r_over_R_max_affected": np.nan,
                    "alpha_min_affected": np.nan,
                    "alpha_max_affected": np.nan,
                    "n_polar_points_affected": 0,
                    "max_Cl_before": float(polar_before["cl"].max()),
                    "max_Cl_after": float(polar_before["cl"].max()),
                    "max_Cd_before": float(polar_before["cd"].max()),
                    "max_Cd_after": float(polar_before["cd"].max()),
                    "max_delta_Cl": 0.0,
                    "max_delta_Cd": 0.0,
                    "alpha_at_max_delta_Cl": np.nan,
                    "max_abs_delta_Cd": 0.0,
                    "post_stall_or_extrapolated_points_affected": 0,
                    "correction_applied_more_than_once": correction_requested_more_than_once,
                    "warning": "",
                })
                continue

            # Find all original blade stations that use this airfoil
            mask  = self.geometry.airfoil_names == airfoil_name
            r_sec = self.geometry.airfoil_r[mask]
            r_over_R_sec = r_sec / self.geometry.R
            radial_mask = (
                (r_over_R_sec >= min_r_over_R)
                & (r_over_R_sec <= max_r_over_R)
            )
            r_sec = r_sec[radial_mask]
            if len(r_sec) == 0:
                diagnostics.append({
                    "airfoil": airfoil_name,
                    "applied": False,
                    "reason": "no_airfoil_stations_in_requested_r_over_R_range",
                    "r_over_R_min_affected": np.nan,
                    "r_over_R_max_affected": np.nan,
                    "alpha_min_affected": np.nan,
                    "alpha_max_affected": np.nan,
                    "n_polar_points_affected": 0,
                    "max_Cl_before": float(polar_before["cl"].max()),
                    "max_Cl_after": float(polar_before["cl"].max()),
                    "max_Cd_before": float(polar_before["cd"].max()),
                    "max_Cd_after": float(polar_before["cd"].max()),
                    "max_delta_Cl": 0.0,
                    "max_delta_Cd": 0.0,
                    "alpha_at_max_delta_Cl": np.nan,
                    "max_abs_delta_Cd": 0.0,
                    "post_stall_or_extrapolated_points_affected": 0,
                    "correction_applied_more_than_once": correction_requested_more_than_once,
                    "warning": "",
                })
                continue

            c_sec    = np.interp(r_sec, self.geometry.r, self.geometry.chord)
            r_mean   = float(np.mean(r_sec))
            c_over_r = float(np.mean(c_sec / r_sec))

            # alpha_0 from the monotone linear region only
            lin_mask = (
                (measured_polar["alpha"] >= -10.0)
                & (measured_polar["alpha"] <= 5.0)
            )
            p_lin = measured_polar[lin_mask].sort_values("alpha")
            alpha_0_est = float(
                np.interp(0.0, p_lin["cl"].values, p_lin["alpha"].values)
            ) if len(p_lin) >= 2 else -2.0

            rc = RotationalCorrection(
                c_over_r    = c_over_r,
                local_tsr   = float(np.mean(omega_ref * r_sec / v_ref)),
                R_over_r    = self.geometry.R / r_mean,
                tip_tsr     = omega_ref * self.geometry.R / v_ref,
                alpha_0_deg = alpha_0_est,
            )

            corrected_source = rc.apply(
                correction_source,
                alpha_min_deg=alpha_min_deg,
                alpha_max_deg=alpha_max_deg,
                delta_cl_max=delta_cl_max,
            )
            corrected_reference = corrected_source.copy()
            if (
                apply_before_viterna
                and self.extrapolate_360
                and self._needs_extrapolation(corrected_reference)
            ):
                corrected_reference = self.viterna.extrapolate(corrected_reference)
            self.polars_corrected_reference[airfoil_name] = corrected_reference

            delta_cl = corrected_source["cl"].values - polar_before["cl"].values
            delta_cd = corrected_source["cd"].values - polar_before["cd"].values
            changed = (np.abs(delta_cl) > 1e-12) | (np.abs(delta_cd) > 1e-12)
            alpha_changed = corrected_source.loc[changed, "alpha"]
            max_delta_idx = int(np.argmax(delta_cl)) if len(delta_cl) else 0

            raw_polar = measured_polar
            n_post_stall_or_extrapolated = 0
            warning = ""
            if raw_polar is not None and len(raw_polar) > 0 and not apply_before_viterna:
                raw_alpha_min = float(raw_polar["alpha"].min())
                raw_alpha_max = float(raw_polar["alpha"].max())
                extrapolated = (
                    (corrected_source["alpha"].values < raw_alpha_min)
                    | (corrected_source["alpha"].values > raw_alpha_max)
                )

                try:
                    alpha_stall_pos, alpha_stall_neg = self.viterna.find_stall_angles(raw_polar)
                    post_stall = (
                        (corrected_source["alpha"].values > alpha_stall_pos)
                        | (corrected_source["alpha"].values < alpha_stall_neg)
                    )
                except Exception:
                    post_stall = np.zeros(len(corrected_source), dtype=bool)

                post_stall_or_extrapolated = extrapolated | post_stall
                n_post_stall_or_extrapolated = int(
                    np.sum(changed & post_stall_or_extrapolated)
                )
                if n_post_stall_or_extrapolated > 0:
                    warning = (
                        "rotational correction affected post-stall and/or "
                        "360-degree extrapolated polar points"
                    )

            diagnostics.append({
                "airfoil": airfoil_name,
                "applied": True,
                "reason": "",
                "r_over_R_min_affected": float(np.min(r_sec / self.geometry.R)),
                "r_over_R_max_affected": float(np.max(r_sec / self.geometry.R)),
                "alpha_min_affected": float(alpha_changed.min()) if changed.any() else np.nan,
                "alpha_max_affected": float(alpha_changed.max()) if changed.any() else np.nan,
                "n_polar_points_affected": int(np.sum(changed)),
                "max_Cl_before": float(polar_before["cl"].max()),
                "max_Cl_after": float(corrected_source["cl"].max()),
                "max_Cd_before": float(polar_before["cd"].max()),
                "max_Cd_after": float(corrected_source["cd"].max()),
                "max_delta_Cl": float(np.max(delta_cl)) if len(delta_cl) else 0.0,
                "max_delta_Cd": float(np.max(np.abs(delta_cd))) if len(delta_cd) else 0.0,
                "alpha_at_max_delta_Cl": float(corrected_source.loc[max_delta_idx, "alpha"]),
                "max_abs_delta_Cd": float(np.max(np.abs(delta_cd))) if len(delta_cd) else 0.0,
                "post_stall_or_extrapolated_points_affected": n_post_stall_or_extrapolated,
                "correction_applied_more_than_once": correction_requested_more_than_once,
                "warning": warning,
            })

        self._build_corrected_polars(
            tsr_ref=tsr_ref,
            v_ref=v_ref,
            min_r_over_R=min_r_over_R,
            max_r_over_R=max_r_over_R,
            alpha_min_deg=alpha_min_deg,
            alpha_max_deg=alpha_max_deg,
            delta_cl_max=delta_cl_max,
            apply_before_viterna=apply_before_viterna,
        )
        self.apply_rotational_correction = True
        self.rotational_correction_applied = True
        self.rotational_correction_diagnostics = diagnostics
        return diagnostics

    def _build_corrected_polars(
        self,
        tsr_ref=None,
        v_ref=None,
        min_r_over_R=0.0,
        max_r_over_R=1.0,
        alpha_min_deg=-25.0,
        alpha_max_deg=25.0,
        delta_cl_max=0.3,
        apply_before_viterna=False,
    ):
        """
        Pre-compute per-section Du-Selig corrected polars at every
        radial station in the geometry.

        Polars are stored in self.polars_corrected keyed by section index
        (integer) so lookups during the BEM sweep are O(1).
        """
        if tsr_ref is not None and v_ref is not None:
            self.V_ref = v_ref
            self.omega_ref = tsr_ref * v_ref / self.geometry.R

        if self.V_ref is None or self.omega_ref is None:
            raise ValueError(
                "V_ref and omega_ref must be set to use rotational correction."
            )

        R       = self.geometry.R
        tip_tsr = self.omega_ref * R / self.V_ref

        # section_polar_map[i] = corrected polar DataFrame for section i
        self.section_polar_map = {}

        for idx, (r, airfoil_name) in enumerate(
            zip(self.geometry.r, self._map_airfoil_per_section())
        ):
            if r <= 0:
                self.section_polar_map[idx] = self.polars_uncorrected_360.get(airfoil_name)
                continue

            chord   = float(self.geometry.chord[idx])
            cr      = chord / r
            Rr      = R / r
            tsr_loc = self.omega_ref * r / self.V_ref

            r_over_R = r / R
            if r_over_R < min_r_over_R or r_over_R > max_r_over_R:
                self.section_polar_map[idx] = self.polars_uncorrected_360[airfoil_name]
                continue

            # estimate the zero lift angle from the linear region [-10, 5] deg
            # using only the monotone portion to avoid np.interp issues
            measured = self.polars_raw[airfoil_name]
            mask = (measured["alpha"] >= -10.0) & (measured["alpha"] <= 5.0)
            p_lin = measured[mask].sort_values("alpha")
            if len(p_lin) >= 2:
                alpha_0_est = float(
                    np.interp(0.0, p_lin["cl"].values, p_lin["alpha"].values)
                )
            else:
                alpha_0_est = -2.0   # fallback

            rc = RotationalCorrection(
                c_over_r    = cr,
                local_tsr   = tsr_loc,
                R_over_r    = Rr,
                tip_tsr     = tip_tsr,
                alpha_0_deg = alpha_0_est,
            )

            correction_source = (
                measured if apply_before_viterna
                else self.polars_uncorrected_360[airfoil_name]
            )
            corrected = rc.apply(
                correction_source,
                alpha_min_deg=alpha_min_deg,
                alpha_max_deg=alpha_max_deg,
                delta_cl_max=delta_cl_max,
            )
            if (
                apply_before_viterna
                and self.extrapolate_360
                and self._needs_extrapolation(corrected)
            ):
                corrected = self.viterna.extrapolate(corrected)
            self.section_polar_map[idx] = corrected
            self.polars_corrected[idx] = corrected

    def get_representative_section_polars(self, airfoil_name, targets=(0.2, 0.3, 0.4)):
        """Return nearest corrected section polars for plotting diagnostics."""
        out = {}
        names = self._map_airfoil_per_section()
        r_over_R = self.geometry.r / self.geometry.R
        for target in targets:
            candidates = [
                idx for idx, name in enumerate(names)
                if name == airfoil_name and idx in self.section_polar_map
            ]
            if not candidates:
                continue
            idx = min(candidates, key=lambda i: abs(r_over_R[i] - target))
            out[float(r_over_R[idx])] = self.section_polar_map[idx]
        return out

    def _map_airfoil_per_section(self):
        """
        Return list of airfoil names, one per BEM section, by finding
        the nearest original airfoil station for each remeshed section.
        """
        names = []
        for r in self.geometry.r:
            idx = np.argmin(np.abs(self.geometry.airfoil_r - r))
            names.append(self.geometry.airfoil_names[idx])
        return names

    def get_coefficients(self, airfoil_name, alpha_deg, r=None, section_idx=None,
                         Re=None):
        """
        Return (Cl, Cd) for a given airfoil name and angle of attack.

        if rotational correction is on and section_idx is given, the
        pre built per section corrected polar is used (O(1) lookup).

        Reynolds dependence: when the airfoil has more than one Reynolds polar
        AND a local Re is given, (Cl, Cd) get interpolated on Re between the two
        bracketing polars (Re clamped to the table range, so it never
        extrapolates). for single Re airfoils, or when Re is None, it is just the
        plain single polar lookup. the rotational correction branch does not care
        about Re (its corrected polars are single Re anyway).
        """
        if (
            self.apply_rotational_correction
            and section_idx is not None
            and hasattr(self, "section_polar_map")
            and section_idx in self.section_polar_map
            and self.section_polar_map[section_idx] is not None
        ):
            polar = self.section_polar_map[section_idx]
            return self._interp_polar(polar, alpha_deg)

        re_table = self.polars_by_re.get(airfoil_name)
        if Re is not None and re_table is not None and len(re_table) > 1:
            return self._interp_polar_reynolds(re_table, alpha_deg, Re)

        polar = self.polars[airfoil_name]
        return self._interp_polar(polar, alpha_deg)

    def _interp_polar(self, polar, alpha_deg):
        """Alpha interpolation of one polar DataFrame -> (Cl, Cd)."""
        alpha_deg = self.wrap_angle(alpha_deg)
        Cl = float(np.interp(alpha_deg, polar["alpha"], polar["cl"]))
        Cd = float(np.interp(alpha_deg, polar["alpha"], polar["cd"]))
        return Cl, Cd

    def _interp_polar_reynolds(self, re_table, alpha_deg, Re):
        """Interpolate (Cl, Cd) on Reynolds number across a Re sorted table.

        re_table is a list of (Re_value, polar_df) sorted by ascending Re. Re is
        clamped to [Re_min, Re_max] so we never extrapolate. at each bracketing
        Re i interpolate the polar on alpha first, then blend the two on Re.
        """
        res = [rv for rv, _ in re_table]
        Re_lo, Re_hi = res[0], res[-1]
        if Re <= Re_lo:
            return self._interp_polar(re_table[0][1], alpha_deg)
        if Re >= Re_hi:
            return self._interp_polar(re_table[-1][1], alpha_deg)
        # Locate bracketing pair.
        for k in range(1, len(re_table)):
            r_lo, p_lo = re_table[k - 1]
            r_hi, p_hi = re_table[k]
            if r_lo <= Re <= r_hi:
                Cl_lo, Cd_lo = self._interp_polar(p_lo, alpha_deg)
                Cl_hi, Cd_hi = self._interp_polar(p_hi, alpha_deg)
                w = (Re - r_lo) / (r_hi - r_lo) if r_hi > r_lo else 0.0
                Cl = (1.0 - w) * Cl_lo + w * Cl_hi
                Cd = (1.0 - w) * Cd_lo + w * Cd_hi
                return float(Cl), float(Cd)
        # Fallback (should not reach): use the highest-Re polar.
        return self._interp_polar(re_table[-1][1], alpha_deg)

    def get_blended_coefficients(self, r, alpha_deg, section_idx=None, Re=None):
        """
        Return (Cl, Cd) at a given radial position with smooth blending
        across airfoil transitions.

        an optional local Reynolds number Re is passed down to the per airfoil
        lookups. it only matters for airfoils with several Re polars, otherwise
        it is ignored.

        the geometry CSV gives one airfoil per station (e.g. cylinder until
        r=0.5, S823 from 0.5 to 1.5, S822 beyond), so the airfoil assignment is a
        step function. to avoid sharp jumps in the radial force distribution this
        method blends between neighbouring named airfoils over a transition zone
        of half width blend_half_width (default 5% of R) centred on each
        boundary. inside that zone the weight follows a smooth cosine ramp (0 at
        one end, 1 at the other).

        Outside the transition zone the section uses purely the named
        airfoil (no blending overhead).
        """
        airfoil_stations = self.geometry.airfoil_names
        r_stations       = self.geometry.airfoil_r

        # Out-of-range guards
        if r <= r_stations[0]:
            return self.get_coefficients(
                airfoil_stations[0], alpha_deg, section_idx=section_idx, Re=Re,
            )
        if r >= r_stations[-1]:
            return self.get_coefficients(
                airfoil_stations[-1], alpha_deg, section_idx=section_idx, Re=Re,
            )

        # Locate the bracketing CSV rows
        upper_index = np.searchsorted(r_stations, r)
        lower_index = upper_index - 1
        airfoil_lo  = airfoil_stations[lower_index]
        airfoil_hi  = airfoil_stations[upper_index]

        # Default behavior: if the two CSV rows share an airfoil name,
        # we still need to check whether r is inside a transition zone
        # centred on a nearby NAME-CHANGE boundary (not just a row gap).
        # Find the nearest name-change boundary in each direction.
        R = self.geometry.R
        blend_half_width = self.blend_half_width_frac * R

        # Walk outward to find the closest boundary on each side
        boundary_left  = None
        boundary_right = None

        for i in range(lower_index, 0, -1):
            if airfoil_stations[i] != airfoil_stations[i - 1]:
                # boundary lies between r_stations[i-1] and r_stations[i]
                boundary_left = 0.5 * (r_stations[i - 1] + r_stations[i])
                af_left_inner = airfoil_stations[i - 1]
                af_left_outer = airfoil_stations[i]
                break

        for i in range(upper_index, len(airfoil_stations) - 1):
            if airfoil_stations[i] != airfoil_stations[i + 1]:
                boundary_right = 0.5 * (r_stations[i] + r_stations[i + 1])
                af_right_inner = airfoil_stations[i]
                af_right_outer = airfoil_stations[i + 1]
                break

        # Distance to the nearest boundary on each side
        dist_left  = (r - boundary_left)  if boundary_left  is not None else np.inf
        dist_right = (boundary_right - r) if boundary_right is not None else np.inf

        if min(dist_left, dist_right) >= blend_half_width:
            # Far from any transition: pure named-airfoil lookup
            return self.get_coefficients(
                airfoil_lo, alpha_deg, section_idx=section_idx, Re=Re,
            )

        # Inside a transition zone: identify which boundary is closer
        if dist_left < dist_right:
            boundary    = boundary_left
            af_inner    = af_left_inner   # airfoil INSIDE the boundary (smaller r)
            af_outer    = af_left_outer
            distance    = dist_left
        else:
            boundary    = boundary_right
            af_inner    = af_right_inner
            af_outer    = af_right_outer
            distance    = -dist_right  # negative: r is INSIDE the boundary

        # Position relative to the boundary, in [-blend_half_width, +blend_half_width]
        s = (r - boundary) / blend_half_width   # -1 to +1 inside zone

        # Cosine ramp: weight of OUTER airfoil grows smoothly from 0 to 1
        # as we cross the boundary
        w_outer = 0.5 * (1.0 + np.cos(np.pi * (1.0 - 0.5 * (s + 1.0))))
        w_outer = float(np.clip(w_outer, 0.0, 1.0))
        w_inner = 1.0 - w_outer

        if af_inner == af_outer:
            return self.get_coefficients(
                af_inner, alpha_deg, section_idx=section_idx, Re=Re,
            )

        Cl_in, Cd_in = self.get_coefficients(
            af_inner, alpha_deg, section_idx=section_idx, Re=Re,
        )
        Cl_out, Cd_out = self.get_coefficients(
            af_outer, alpha_deg, section_idx=section_idx, Re=Re,
        )

        Cl = w_inner * Cl_in + w_outer * Cl_out
        Cd = w_inner * Cd_in + w_outer * Cd_out
        return Cl, Cd

    def wrap_angle(self, alpha_deg):
        return ((alpha_deg + 180) % 360) - 180
