import numpy as np
from corrections import BEMCorrections


class BEMSolver:
    def __init__(
        self,
        geometry,
        aero,
        rho=1.225,
        max_iter=200,
        tol=1e-5,
        relaxation=0.3,
        use_tip_loss=True,
        use_root_loss=True,
        use_glauert=True,
        use_root_cutoff=False,
        r_root=None,
        root_cutoff_fraction=0.15,
        high_induction_model="buhl",
        debug=False,
        record_iteration_history=False,
        buhl_transition_model="smooth_blend",
        buhl_deactivation_count_required=3,
        drag_in_induction=True,
        solver_method="picard",
    ):
        self.geometry = geometry
        self.aero = aero
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.relaxation = relaxation
        self.debug = debug
        self.record_iteration_history = record_iteration_history

        # which section solver to use:
        #   "picard" : multi start fixed point iteration (the default one)
        #   "ning"   : Ning (2014) single residual in phi, solved with a
        #              bracketed Brent. converges no matter what once we have a
        #              sign change in the bracket. use it for stalled blades.
        if solver_method not in ("picard", "ning"):
            raise ValueError(
                f"solver_method must be 'picard' or 'ning', got {solver_method!r}"
            )
        self.solver_method = solver_method

        self.corrections = BEMCorrections(
            B=self.geometry.B,
            R=self.geometry.R,
            r_root=r_root if r_root is not None else self.geometry.r_hub,
            min_F=1e-4,
            a_min=-0.2,
            a_max=0.95,   # Glauert correction can produce values up to ~0.95
            a_prime_min=-0.5,
            a_prime_max=0.5,
            relaxation_a=relaxation,
            relaxation_ap=relaxation,
            use_tip_loss=use_tip_loss,
            use_root_loss=use_root_loss,
            use_glauert=use_glauert,
            use_root_cutoff=use_root_cutoff,
            root_cutoff_fraction=root_cutoff_fraction,
            high_induction_model=high_induction_model,
            buhl_transition_model=buhl_transition_model,
            buhl_deactivation_count_required=buhl_deactivation_count_required,
            drag_in_induction=drag_in_induction,
        )

    def solve_section(self, r, chord, twist, V_inf, omega, pitch=0.0, section_idx=None):
        # try two starting guesses and keep the one with the lower residual.
        #
        # the reason i do this: in the Glauert regime the fixed point has two
        # stable branches. the real one has small a (around 0.1 to 0.3). the
        # fake one sits clamped at a_max (~0.95), which pushes phi to 0, alpha
        # negative, Cl to 0, and then the iteration "converges" for free because
        # two clamped values in a row are identical (|a_new - a_old| = 0 < tol).
        #
        # the two starts:
        #     a = 0.0  always safe, walks up to the physical solution
        #     a = 1/3  Betz value, good for loaded sections at low TSR
        #
        # winner is whoever has the smallest final residual |a_new - a_final|.
        # a clamped solution has a big residual, a real one has ~0.

        a_max = self.corrections.a_max  # typically 0.95

        def _run_from(a_init, start_label):
            a          = a_init
            a_prime    = 0.0
            glauert_active = False
            buhl_deactivation_counter = 0
            converged  = False
            last_corr  = {}
            iteration_history = []

            for it in range(self.max_iter):
                a_old      = a
                a_prime_old = a_prime

                V_axial      = V_inf * (1.0 - a)
                V_tangential = omega * r * (1.0 + a_prime)
                phi          = np.arctan2(V_axial, V_tangential)
                alpha        = phi - twist - pitch

                # local Reynolds from the current relative velocity (stays None
                # unless we actually loaded several polars per airfoil)
                Re_local = self._section_reynolds(V_axial, V_tangential, chord)
                Cl, Cd = self.aero.get_blended_coefficients(
                    r, np.degrees(alpha), section_idx=section_idx, Re=Re_local)

                a, a_prime, corr_info = self.corrections.update_induction_factors(
                    r=r, chord=chord, phi=phi, Cl=Cl, Cd=Cd,
                    a_old=a_old, a_prime_old=a_prime_old,
                    glauert_active_prev=glauert_active,
                    buhl_deactivation_counter=buhl_deactivation_counter,
                )
                glauert_active = corr_info["glauert_active"]
                buhl_deactivation_counter = corr_info.get(
                    "buhl_deactivation_counter", buhl_deactivation_counter
                )
                last_corr      = corr_info

                # true residual is how far the unrelaxed update would move,
                # |a_raw - a_old|. if i test the relaxed step |a - a_old| instead
                # it equals relaxation * |a_raw - a_old|, so a heavily relaxed run
                # could pass the tolerance without the fixed point being solved.
                # so test the raw one.
                a_raw_it  = corr_info.get("a_raw", a)
                ap_raw_it = corr_info.get("a_prime_raw", a_prime)
                delta_a  = abs(a_raw_it  - a_old)
                delta_ap = abs(ap_raw_it - a_prime_old)

                if self.record_iteration_history:
                    iteration_history.append({
                        "start_label": start_label,
                        "iteration": it + 1,
                        "r": r,
                        "r_R": r / self.geometry.R,
                        "a_old": a_old,
                        "a_new": corr_info.get("a_raw", np.nan),
                        "a_relaxed": a,
                        "a_prime_old": a_prime_old,
                        "a_prime_new": corr_info.get("a_prime_raw", np.nan),
                        "a_prime_relaxed": a_prime,
                        "phi_deg": np.degrees(phi),
                        "alpha_deg": np.degrees(alpha),
                        "Cl": Cl,
                        "Cd": Cd,
                        "Cn": corr_info.get("Cn", np.nan),
                        "Ct": corr_info.get("Ct", np.nan),
                        "K": corr_info.get("K", np.nan),
                        "CT_local": corr_info.get("CT_local", np.nan),
                        "CT_local_over_F": corr_info.get("CT_local_over_F", np.nan),
                        "F": corr_info.get("F", np.nan),
                        "residual_a": delta_a,
                        "residual_a_prime": delta_ap,
                        "high_induction_active": corr_info.get("glauert_active", False),
                        "buhl_transition_model": corr_info.get(
                            "buhl_transition_model", ""
                        ),
                        "buhl_blend_factor": corr_info.get(
                            "buhl_blend_factor", np.nan
                        ),
                        "buhl_latched": corr_info.get("buhl_latched", False),
                        "buhl_deactivation_counter": corr_info.get(
                            "buhl_deactivation_counter", np.nan
                        ),
                        "a_momentum": corr_info.get("a_momentum", np.nan),
                        "a_buhl": corr_info.get("a_buhl", np.nan),
                    })

                # dont let the a_max clamp fake a convergence. when a is pinned
                # at the cap two iterates are equal and pass the tol test for
                # nothing, so only accept it if a is not sitting on the ceiling.
                at_cap = (a >= a_max - 1e-6)

                if delta_a < self.tol and delta_ap < self.tol and not at_cap:
                    converged = True
                    break

            last_corr["iterations"] = it + 1

            # Residual: how far is the final raw (pre-clamp) a from a_final?
            # A clamped solution has large residual; a true solution has ~0.
            a_raw = last_corr.get("a_raw", a)
            residual = abs(a_raw - a)

            return a, a_prime, converged, last_corr, residual, iteration_history

        if self.solver_method == "ning":
            # Ning (2014) bracketed phi residual solve. it reuses the same
            # corrections as Picard (axial/tangential induction, drag toggle,
            # smooth_blend high thrust), so same equations, but it always
            # converges on heavily loaded sections.
            (best_a, best_ap, converged,
             last_corr_info, best_history) = self._ning_solve(
                r, chord, twist, V_inf, omega, pitch, section_idx)
        else:
            # always start from a=0 too, not only from the Betz guess. a=0 cant
            # get pulled into the fake high-a branch.
            best_a, best_ap, converged, last_corr_info, best_res, best_history = _run_from(
                0.0, "a0"
            )
            a2, ap2, conv2, corr2, res2, hist2 = _run_from(1.0 / 3.0, "a_one_third")

            # keep the start with the smaller residual. in practice a=0 wins at
            # high TSR / light loading, and 1/3 wins at low TSR (stalled root).
            if res2 < best_res:
                best_a, best_ap, converged, last_corr_info = a2, ap2, conv2, corr2
                best_history = hist2

        a       = best_a
        a_prime = best_ap
        numerically_converged = converged
        discriminant_clipped = bool(
            last_corr_info.get("discriminant_clipped", False)
        )
        physically_valid = not discriminant_clipped
        invalid_reason = (
            "negative_reference_glauert_discriminant"
            if discriminant_clipped else ""
        )
        converged = numerically_converged and physically_valid

    
        # --------------------------------------------------
        # Recompute final values from converged a and a'
        # Forces are always computed from fresh aerodynamic lookup,
        # not from last_corr_info, so they are correct even when
        # the section hit max_iter without converging.
        # --------------------------------------------------

        V_axial      = V_inf * (1.0 - a)
        V_tangential = omega * r * (1.0 + a_prime)
        phi          = np.arctan2(V_axial, V_tangential)
        alpha        = phi - twist - pitch
        alpha_deg    = np.degrees(alpha)

        # local Reynolds from the converged relative velocity (None unless
        # multi Re polars are loaded)
        Re_local = self._section_reynolds(V_axial, V_tangential, chord)
        Cl, Cd = self.aero.get_blended_coefficients(
            r, alpha_deg, section_idx=section_idx, Re=Re_local)

        Cn    = self.corrections.normal_force_coefficient(Cl, Cd, phi)
        Ct    = self.corrections.tangential_force_coefficient(Cl, Cd, phi)
        sigma = self.corrections.local_solidity(self.geometry.B, chord, r)
        F     = self.corrections.prandtl_loss_factor(r, phi)
        CT_local = self.corrections.local_thrust_coefficient(sigma, Cn, phi)

        W = np.sqrt(V_axial ** 2 + V_tangential ** 2)
        q = 0.5 * self.rho * W ** 2

        dFn = q * chord * Cn
        dFt = q * chord * Ct

        root_weight = self.corrections.root_weight(r)

        if self.debug:
            print(
                f"[BEM] r={r:.3f}m"
                f"  model={self.corrections.high_induction_model}"
                f"  K={last_corr_info.get('K', float('nan')):.4f}"
                f"  fg={last_corr_info.get('fg', float('nan')):.4f}"
                f"  a_old={last_corr_info.get('a_old', float('nan')):.4f}"
                f"  a_raw={last_corr_info.get('a_raw', float('nan')):.4f}"
                f"  a={a:.4f}"
                f"  glauert={last_corr_info.get('glauert_active', False)}"
                f"  conv={converged}"
            )

        return {
            "r": r,
            "chord": chord,
            "twist": twist,
            "phi": phi,
            "alpha": alpha,
            "alpha_deg": alpha_deg,
            "a": a,
            "a_prime": a_prime,
            "Cl": Cl,
            "Cd": Cd,
            "Cn": Cn,
            "Ct": Ct,
            "CT_local": CT_local,
            "CT_local_over_F": last_corr_info.get("CT_local_over_F", np.nan),
            "CT_critical": last_corr_info.get("CT_critical", np.nan),
            "glauert_active": last_corr_info.get("glauert_active", False),
            "high_induction_active": last_corr_info.get("glauert_active", False),
            "buhl_transition_model": last_corr_info.get("buhl_transition_model", ""),
            "buhl_blend_factor": last_corr_info.get("buhl_blend_factor", np.nan),
            "buhl_latched": last_corr_info.get("buhl_latched", False),
            "buhl_deactivation_counter": last_corr_info.get(
                "buhl_deactivation_counter", np.nan
            ),
            "a_momentum": last_corr_info.get("a_momentum", np.nan),
            "a_buhl": last_corr_info.get("a_buhl", np.nan),
            "sigma": sigma,
            "F": F,
            "K": last_corr_info.get("K", np.nan),
            "fg": last_corr_info.get("fg", np.nan),
            "raw_discriminant": last_corr_info.get("raw_discriminant", np.nan),
            "clipped_discriminant": last_corr_info.get("clipped_discriminant", np.nan),
            "discriminant": last_corr_info.get("raw_discriminant", np.nan),
            "discriminant_clipped": discriminant_clipped,
            "a_old": last_corr_info.get("a_old", np.nan),
            "a_raw": last_corr_info.get("a_raw", np.nan),
            "a_final": last_corr_info.get("a_final", np.nan),
            "a_prime_raw": last_corr_info.get("a_prime_raw", np.nan),
            "high_induction_model": last_corr_info.get("high_induction_model", ""),
            "iterations": last_corr_info.get("iterations", np.nan),
            "numerically_converged": numerically_converged,
            "physically_valid": physically_valid,
            "invalid_reason": invalid_reason,
            "root_weight": root_weight,
            "W": W,
            "dFn": dFn,
            "dFt": dFt,
            "converged": converged,
            "iteration_history": best_history if self.record_iteration_history else [],
            "CT_critical_final": last_corr_info.get("CT_critical", np.nan),
            "glauert_active_final": last_corr_info.get("glauert_active", False),
        }

    def _section_reynolds(self, V_axial, V_tangential, chord):
        """Local section Reynolds number for Re dependent polars, or None.

        returns None (so nothing changes) unless the aero layer is actually
        carrying multi Re polars. that way the normal single Re case skips the
        hypot/division. when it is active Re = W c / nu, with W the local
        relative velocity |(V_axial, V_tangential)|.
        """
        if not getattr(self.aero, "reynolds_dependent", False):
            return None
        W = float(np.hypot(V_axial, V_tangential))
        return self.aero.local_reynolds(W, chord)

    def _ning_solve(self, r, chord, twist, V_inf, omega, pitch, section_idx):
        """Ning (2014) bracketed phi residual section solve.

        turns the coupled (a, a') fixed point into a single residual in the
        inflow angle phi:

            R(phi) = sin(phi)/(1 - a(phi))
                     - cos(phi)/(lambda_r * (1 + a'(phi)))

        with lambda_r = omega r / V_inf and a(phi), a'(phi) coming from the same
        corrections the Picard path uses (axial/tangential induction, the drag
        toggle, the history free smooth_blend high thrust model). R changes sign
        across the windmill bracket (0, pi/2), so Brent finds the root every time
        once it has a sign changing sub interval. that is why it holds up on the
        heavily loaded sections where Picard oscillates or gets stuck.

        returns (a, a_prime, converged, corr_info, history), same tuple the
        Picard branch feeds into the shared force recompute.
        """
        from scipy.optimize import brentq

        B = self.geometry.B
        sigma = self.corrections.local_solidity(B, chord, r)
        lambda_r = (omega * r / V_inf) if V_inf != 0.0 else 0.0
        n_eval = [0]

        def _inductions_at(phi, Re):
            alpha = phi - twist - pitch
            Cl, Cd = self.aero.get_blended_coefficients(
                r, np.degrees(alpha), section_idx=section_idx, Re=Re)
            Ct = self.corrections.tangential_force_coefficient(Cl, Cd, phi)
            # drop drag from the induction Cn if that is how it was set up
            if self.corrections.drag_in_induction:
                Cn_ind = self.corrections.normal_force_coefficient(Cl, Cd, phi)
            else:
                Cn_ind = Cl * np.cos(phi)
            F = self.corrections.prandtl_loss_factor(r, phi)
            a, _, _ = self.corrections.axial_induction(sigma, Cn_ind, phi, F)
            a_prime = self.corrections.tangential_induction(sigma, Ct, phi, F)
            return a, a_prime, Cl, Cd

        def _inductions(phi):
            # first pass at Re=None
            a, a_prime, Cl, Cd = _inductions_at(phi, None)
            # one extra self consistent Reynolds step. skipped completely unless
            # multi Re polars are loaded.
            if getattr(self.aero, "reynolds_dependent", False):
                W = np.hypot(V_inf * (1.0 - a), omega * r * (1.0 + a_prime))
                Re = self.aero.local_reynolds(W, chord)
                a, a_prime, Cl, Cd = _inductions_at(phi, Re)
            return a, a_prime, Cl, Cd

        def residual(phi):
            n_eval[0] += 1
            a, a_prime, _, _ = _inductions(phi)
            denom_t = lambda_r * (1.0 + a_prime)
            if abs(1.0 - a) < 1e-12 or abs(denom_t) < 1e-12:
                # Degenerate; nudge to keep the residual finite and signed.
                return np.sin(phi) - np.cos(phi)
            return np.sin(phi) / (1.0 - a) - np.cos(phi) / denom_t

        # windmill bracket (0, pi/2): scan for sign changes and take the root at
        # the HIGHEST phi. why the highest one: near the root alpha = phi - twist
        # - pitch goes very negative, and there the blended polars throw Cl
        # around and slam a into its clamps, which makes fake low phi crossings.
        # those are just airfoil table artifacts, not real operating points. the
        # real windmill solution is always the highest phi crossing, where alpha
        # is in the normal range and a is small and smooth. so basically i go
        # down from phi=pi/2 (R>0 there) and Brent the first sign change i hit.
        phi_lo, phi_hi = 1e-6, np.pi / 2.0 - 1e-6
        n_scan = 60
        phis = np.linspace(phi_lo, phi_hi, n_scan)
        fvals = np.array([residual(p) for p in phis], dtype=float)
        phi_star = None
        converged = False
        for k in range(n_scan - 1, 0, -1):  # high phi -> low phi
            f_hi, f_lo = fvals[k], fvals[k - 1]
            if np.isfinite(f_hi) and np.isfinite(f_lo) and f_lo * f_hi < 0.0:
                phi_star = brentq(
                    residual, phis[k - 1], phis[k],
                    xtol=1e-10, rtol=8.9e-16, maxiter=self.max_iter)
                converged = True
                break

        if phi_star is None:
            # No sign change in the windmill bracket: report the least-residual
            # phi and flag the section non-converged (do not crash).
            phi_star = float(phis[int(np.nanargmin(np.abs(fvals)))])
            converged = False

        # Build a complete corr_info at the solution. Seeding a_old/a_prime_old
        # with the converged inductions makes relaxation an identity, so the
        # returned a, a' equal the Ning solution while corr_info is fully
        # populated for diagnostics and the shared force recompute.
        a_seed, ap_seed, Cl, Cd = _inductions(phi_star)
        a, a_prime, corr_info = self.corrections.update_induction_factors(
            r=r, chord=chord, phi=phi_star, Cl=Cl, Cd=Cd,
            a_old=a_seed, a_prime_old=ap_seed,
        )
        corr_info["iterations"] = n_eval[0]
        return a, a_prime, converged, corr_info, []

    def solve_rotor(self, V_inf, omega, pitch=0.0):
        results = []
    
        for i in range(self.geometry.n()):
            section = self.geometry.get_section(i)
    
            result = self.solve_section(
                r=section["r"],
                chord=section["chord"],
                twist=section["twist"],
                V_inf=V_inf,
                omega=omega,
                pitch=pitch,
                section_idx=i,
            )
            results.append(result)
    
        r_arr = np.array([res["r"] for res in results])
        dFn = np.array([res["dFn"] for res in results])
        dFt = np.array([res["dFt"] for res in results])
    
        B = self.geometry.B
        R = self.geometry.R   # use tip radius, not last section midpoint
        area = np.pi * R ** 2

        # Primary integration: full-domain annulus (midpoint) rule.
        # The section nodes are annulus centres; weighting each by its annulus
        # width dr integrates the entire [r_hub, R] span, including the root and
        # tip half-annuli that a trapezoid over the centre nodes alone drops.
        if len(r_arr) >= 2:
            edges = np.empty(len(r_arr) + 1)
            edges[1:-1] = 0.5 * (r_arr[:-1] + r_arr[1:])
            edges[0] = max(
                self.geometry.r_hub,
                r_arr[0] - 0.5 * (r_arr[1] - r_arr[0]),
            )
            edges[-1] = min(
                R,
                r_arr[-1] + 0.5 * (r_arr[-1] - r_arr[-2]),
            )
            dr = np.diff(edges)
            thrust = B * np.sum(dFn * dr)
            torque = B * np.sum(dFt * r_arr * dr)
        else:
            # Degenerate single-section case: fall back to trapezoid.
            thrust = B * np.trapezoid(dFn, r_arr)
            torque = B * np.trapezoid(dFt * r_arr, r_arr)

        power = torque * omega
        Cp = power / (0.5 * self.rho * area * V_inf ** 3)
        CT_rotor = thrust / (0.5 * self.rho * area * V_inf ** 2)

        # Diagnostic: trapezoid over section midpoints (the previous primary
        # method). Retained to monitor integration sensitivity; not used for
        # the reported outputs.
        thrust_trapz = B * np.trapezoid(dFn, r_arr)
        torque_trapz = B * np.trapezoid(dFt * r_arr, r_arr)
        power_trapz = torque_trapz * omega
        Cp_trapz = power_trapz / (0.5 * self.rho * area * V_inf ** 3)
        Ct_trapz = thrust_trapz / (0.5 * self.rho * area * V_inf ** 2)

        integration_diagnostics = {
            "method_primary": "annulus_midpoint_full_domain",
            "method_diagnostic": "trapezoid_over_section_midpoints",
            "thrust_trapz": thrust_trapz,
            "torque_trapz": torque_trapz,
            "power_trapz": power_trapz,
            "Cp_trapz": Cp_trapz,
            "Ct_trapz": Ct_trapz,
            "delta_Cp_trapz_minus_primary": Cp_trapz - Cp,
            "delta_Ct_trapz_minus_primary": Ct_trapz - CT_rotor,
        }

        # rotor level convergence report. just diagnostics, it sums up the per
        # section state we already have and does not feed back into anything
        # integrated above.
        n_sec = len(results)
        sec_conv = np.array(
            [bool(s.get("converged", False)) for s in results], dtype=bool)
        sec_num_conv = np.array(
            [bool(s.get("numerically_converged", False)) for s in results], dtype=bool)
        sec_phys = np.array(
            [bool(s.get("physically_valid", True)) for s in results], dtype=bool)
        # True fixed-point residual |a_raw - a| of the *reported* solution per
        # section (NaN where a_raw is unavailable). This is the local
        # momentum/blade-element balance residual that convergence tests against.
        fp_res = np.array(
            [abs(float(s.get("a_raw", np.nan)) - float(s.get("a", np.nan)))
             for s in results], dtype=float)
        res_conv = fp_res[sec_conv] if sec_conv.any() else np.array([])
        iters = np.array(
            [float(s.get("iterations", np.nan)) for s in results], dtype=float)

        convergence_report = {
            "n_sections": int(n_sec),
            "n_converged": int(sec_conv.sum()),
            "n_nonconverged": int((~sec_conv).sum()),
            "n_numerically_nonconverged": int((~sec_num_conv).sum()),
            "n_phys_invalid": int((~sec_phys).sum()),
            "all_converged": bool(sec_conv.all()) if n_sec else False,
            "fraction_converged": (
                float(sec_conv.mean()) if n_sec else float("nan")),
            "nonconverged_indices": [int(i) for i in np.nonzero(~sec_conv)[0]],
            "max_fp_residual": (
                float(np.nanmax(fp_res)) if np.isfinite(fp_res).any() else None),
            "max_fp_residual_converged": (
                float(np.nanmax(res_conv))
                if res_conv.size and np.isfinite(res_conv).any() else None),
            "max_iterations": (
                int(np.nanmax(iters)) if np.isfinite(iters).any() else 0),
            "tol": float(self.tol),
        }

        return {
            "sections": results,
            "thrust":   thrust,
            "torque":   torque,
            "power":    power,
            "Cp":       Cp,
            "Ct":       CT_rotor,
            "integration_diagnostics": integration_diagnostics,
            "convergence_report": convergence_report,
        }
