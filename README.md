# Wind-Turbine BEM Performance Toolkit

A research-grade **Blade Element Momentum (BEM)** solver for horizontal-axis wind
turbines, written from scratch in Python. It takes a blade geometry plus airfoil
polars and produces the full aerodynamic performance picture — Cp(λ, β)
surfaces, power and thrust curves, spanwise loads, and a controller-driven
annual energy estimate.

It has been **benchmarked against an independent reference BEM**. The agreement
is good at high wind (power within ~4% for V ≥ 11 m/s) but **not uniform**: in
the mid-wind range the model under-predicts power and torque, it under-predicts
thrust across the whole envelope, and the reference Cp appears to use a different
normalization. The discrepancies and their likely causes are documented in full
below — see [Validation & known accuracy limitations](#validation--known-accuracy-limitations).
Treat this as a capable, transparent BEM implementation, not a black box with a
single accuracy number.

This is the aerodynamic engine behind a smart (bend–twist-coupled) small wind
turbine study; the same code supported a peer-reviewed Q1 publication on
fluid–structure interaction of an adaptive blade.

## What makes this more than a textbook BEM

The hard part of BEM is not the momentum balance — it is staying physical and
convergent on heavily loaded and stalled sections. This solver implements the
corrections that production codes rely on:

- **Two independent section solvers**, selectable per run:
  - *Picard* multi-start fixed-point iteration that launches from both `a = 0`
    and the Betz value `a = 1/3` and keeps the lower-residual root — this
    rejects the spurious high-induction branch where the iteration falsely
    "converges" against the `a_max` clamp.
  - *Ning (2014)* single-residual-in-φ formulation solved with bracketed Brent
    root-finding for **guaranteed convergence** on stalled blades.
- **Prandtl tip- and root-loss** factors.
- **High-induction (turbulent-wake) correction** via the Buhl empirical model
  with a smooth blend into momentum theory (no discontinuity at the transition).
- **Viterna 360° extrapolation** of measured polars into deep stall.
- **Du–Selig rotational-augmentation** (stall-delay) correction, optional.
- **Reynolds-dependent polars**: supply several polars per airfoil and the
  coefficient lookup interpolates on each section's local Reynolds number.
- **Airfoil blending** across span transitions so cl/cd vary smoothly between
  named airfoils.
- **Full-domain annulus integration** (not just a trapezoid over section
  midpoints) so the root and tip half-annuli are not dropped.
- **Variable-speed / pitch-to-feather controller** (MPPT → rated-speed → rated-
  power regions) driving the power curve and AEP.
- Optional **unsteady BEM** and **dynamic-inflow** (Øye) models for transient
  response.

## Validation & known accuracy limitations

The model was compared on the "Smart Blade" (3-blade, R = 2.5 m, S822/S823
airfoils) against an independent reference BEM at **fixed 193 rpm, pitch 0°**,
using multi-Reynolds ±180° polars. The honest summary: **agreement is wind-speed
dependent, and there are systematic discrepancies that are not yet resolved.**
The raw numbers are committed in
[`validation_table.csv`](Outputs/Smart_Blade_Validated/validation_table.csv);
nothing is hidden.

![Validation vs reference](Outputs/Smart_Blade_Validated/validation_vs_reference.png)

### Error vs the reference, by region (rigid-blade reference)

| Wind speed | Power error | Torque error | Thrust error |
|---|---|---|---|
| < 4 m/s (below cut-in) | meaningless (both ≈ 0, errors > 100%) | meaningless | −38 to −90% |
| 4.5 – 10 m/s (mid wind) | **−15% to −40%** (under-predicts) | −16% to −41% | −24% to −35% |
| 11 – 20 m/s (high wind) | ±0.3% to ±4.4% | ±0.7% to −5.4% | **−20% to −40%** |

A single "median ≈ 4%" figure for power is **misleading**: the median is pulled
down by the many closely-spaced high-wind points. Over the full operating range
the **mean absolute power error is ~14%**, driven by the mid-wind under-prediction.

### Known issues, stated plainly

1. **Mid-wind power/torque under-prediction (15–40%).** Around peak loading
   (V ≈ 5–8 m/s) the model produces noticeably less power than the reference.
   The most likely contributor is that the **Du–Selig rotational/stall-delay
   correction is disabled** in this validated run
   (`apply_rotational_correction = False`); rotational augmentation raises
   inboard lift exactly where the gap is largest. This has not been confirmed by
   re-running with the correction on.
2. **Thrust is under-predicted by 20–40% everywhere**, worsening with wind speed.
   This is the largest and most consistent discrepancy and is **not explained**
   by the power agreement at high wind — i.e. the model can land near the right
   power while still loading the rotor too lightly in the axial direction. Root
   cause is still open.
3. **The Cp comparison is not like-for-like.** At matched wind speed the
   reference's implied ½·ρ·A·V³ is roughly **twice** the model's (the reference
   Cp is ~2× the model's while the dimensional powers nearly match at high wind).
   That points to a different reference area or air density in the reference's Cp
   definition, so the Cp-vs-Cp percentages in the table should be ignored — only
   the **dimensional** P, Q, T are comparable, and even those carry this caveat.
4. **High-wind power agreement may be partly coincidental** given (3): matching a
   dimensional power while the underlying normalization differs is weaker
   evidence than it looks.

### What the comparison does support

The solver is **numerically healthy** — all blade sections converge, the Cp(λ)
curve and Cp,max (≈ 0.45 at λ ≈ 7.75) are physically reasonable for this airfoil
family, and the high-wind power trend tracks the reference. It is a solid,
inspectable BEM implementation; it is **not** yet a quantitatively closed
validation, and the thrust and mid-wind gaps above should be resolved before
the numbers are used for design decisions.

### Design-point summary (model's own output)

| | |
|---|---|
| Rotor radius / blades | 2.5 m / 3 |
| λ optimum / Cp,max | 7.75 / 0.449 |
| Rated rotor speed | 193 rpm |
| Aerodynamic rated power | ~5.0 kW |
| AEP (aero, Weibull c=8, k=2) | 17.9 MWh/yr |

These are self-consistent model outputs, not reference-validated figures.

## Selected results

| Cp(λ, β) surface | Cp & Ct vs λ | Power & thrust curve |
|---|---|---|
| ![cp surface](Outputs/Smart_Blade_Validated/cp_surface.png) | ![cp ct](Outputs/Smart_Blade_Validated/cp_ct_curves.png) | ![power thrust](Outputs/Smart_Blade_Validated/power_thrust_curve.png) |

| Spanwise loads at design point | S822 polar (raw + Viterna 360°) | Blade geometry |
|---|---|---|
| ![sections](Outputs/Smart_Blade_Validated/sections_design_point.png) | ![polar](Outputs/Smart_Blade_Validated/polar_S822.png) | ![geometry](Outputs/Smart_Blade_Validated/blade_geometry.png) |

## Run it

```bash
pip install -r requirements.txt

python MAIN_simple.py                       # geometry, polars, Cp surfaces, pitch sweeps
python MAIN_performance_comp_validated.py   # full performance + validation vs reference
```

Outputs (figures + CSVs) are written to `Outputs/<case_name>/`. Each `MAIN_*`
script has a clearly marked **SETTINGS** block at the top — change the geometry
file, polars, controller limits, or sweep ranges there; everything below is
generic.

## Project layout

```
BEM.py                  BEMSolver: per-section solve (Picard + Ning) and rotor integration
corrections.py          BEMCorrections: tip/root loss, Glauert/Buhl high-induction, induction factors
Aero.py                 polar loading, Reynolds interpolation, airfoil blending
Viterna.py              360-degree polar extrapolation
rotational_correction.py  Du-Selig stall-delay correction
Geometria.py            BladeGeometry: CSV blade definition + spanwise meshing
controller.py           variable-speed / pitch-to-feather controller
unsteady_bem.py         unsteady BEM wrapper
dynamic_inflow.py       Oye dynamic-inflow model

MAIN_simple.py                      compact driver: geometry, polars, Cp surfaces
MAIN_performance.py                 full performance assessment
MAIN_performance_comp.py            performance + high-resolution Cp surface
MAIN_performance_comp_validated.py  performance + validation against reference BEM
MAIN_modelica.py                    export for Modelica/system coupling

Inputs/                 blade geometry + airfoil polars (smart, comercial, multi-Re)
Outputs/                generated figures and CSVs
```

## License

MIT — see [LICENSE](LICENSE).
