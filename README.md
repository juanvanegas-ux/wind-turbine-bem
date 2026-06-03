# Wind Turbine BEM Performance Toolkit

A Blade Element Momentum (BEM) solver for horizontal axis wind turbines, written
from scratch in Python. You give it a blade geometry plus the airfoil polars and
it gives back the full aerodynamic performance picture: Cp(lambda, beta)
surfaces, power and thrust curves, spanwise loads, and a controller driven annual
energy estimate.

I compared it against an independent reference BEM. The agreement is good at high
wind (power within about 4% for V over 11 m/s) but it is not uniform: in the mid
wind range the model under predict power and torque, the thrust is too low across
all the envelope, and the reference Cp looks like it use a different
normalization. All the discrepancies and the probable causes are written down in
the validation section below, nothing is hidden. So please read this like a
capable and transparent BEM implementation, not like a black box with one magic
accuracy number.

This is the aerodynamic engine behind a smart (bend twist coupled) small wind
turbine study, and the same code supported a peer reviewed Q1 publication about
the fluid structure interaction of an adaptive blade.

## What makes it more than a textbook BEM

The hard part of BEM is not the momentum balance, is to stay physical and
convergent on the heavily loaded and stalled sections. This solver implement the
corrections that the production codes rely on:

* Two independent section solvers, you pick one per run:
  * Picard multi start fixed point iteration that launch from both a = 0 and the
    Betz value a = 1/3 and keeps the root with the lower residual. This rejects
    the spurious high induction branch where the iteration "converge" falsely
    against the a_max clamp.
  * Ning (2014) single residual in phi formulation, solved with a bracketed
    Brent root finding for guaranteed convergence on stalled blades.
* Prandtl tip and root loss factors.
* High induction (turbulent wake) correction with the Buhl empirical model,
  blended smoothly into momentum theory so there is no discontinuity in the
  transition.
* Viterna 360 degree extrapolation of the measured polars into deep stall.
* Du Selig rotational augmentation (stall delay) correction, optional.
* Reynolds dependent polars: you give several polars per airfoil and the lookup
  interpolate on the local Reynolds number of each section.
* Airfoil blending across the span transitions, so cl and cd change smoothly
  between named airfoils.
* Full domain annulus integration (not only a trapezoid over the section
  midpoints) so the root and tip half annulus are not dropped.
* Variable speed and pitch to feather controller (MPPT, then rated speed, then
  rated power regions) that drives the power curve and the AEP.
* Optional unsteady BEM and dynamic inflow (Oye) models for the transient
  response.

## Validation and the accuracy problems I know about

The model was compared on the "Smart Blade" (3 blades, R = 2.5 m, S822 and S823
airfoils) against an independent reference BEM at fixed 193 rpm and pitch 0
degrees, using multi Reynolds polars that cover the full plus minus 180 degree
range. The honest summary is that the agreement depends of the wind speed and
there are systematic differences that i did not close yet. The raw numbers are
committed in Outputs/Smart_Blade_Validated/validation_table.csv so anybody can
check.

![Validation vs reference](Outputs/Smart_Blade_Validated/validation_vs_reference.png)

Error against the reference, by region (rigid blade reference):

```
below cut in (under 4 m/s) : power meaningless (both almost zero), thrust 40 to 90% low
mid wind (4.5 to 10 m/s)   : power and torque 15 to 40% low, thrust 24 to 35% low
high wind (11 to 20 m/s)   : power and torque within about 5%, thrust still 20 to 40% low
```

One single "median around 4%" number for the power is misleading, the median is
pulled down by the many high wind points that sit close together. Over the full
operating range the mean absolute power error is closer to 14%, and it comes from
the mid wind under prediction.

The issues, said plainly:

1. Mid wind power and torque is too low (15 to 40%). Around the peak loading
   (V about 5 to 8 m/s) the model make less power than the reference. The most
   probable cause is that the Du Selig rotational / stall delay correction is
   turned off in this validated run (apply_rotational_correction = False), and
   that correction raise the inboard lift exactly where the gap is the biggest.
   I did not confirm it yet by re running with the correction on.
2. Thrust is too low by 20 to 40% everywhere, and it get worse with the wind
   speed. This is the biggest and most consistent difference, and it is not
   explained by the power match at high wind: the model can land near the right
   power while still loading the rotor too light in the axial direction. The root
   cause is still open.
3. The Cp comparison is not like for like. At the same wind speed the reference
   1/2 rho A V^3 is about twice the model one (the reference Cp is around 2x the
   model Cp while the dimensional powers almost match at high wind). That point
   to a different reference area or air density in the reference Cp definition, so
   the Cp vs Cp percentages should be ignored, only the dimensional P, Q and T
   are comparable, and even those carry this caveat.
4. Because of point 3, the high wind power match could be partly a coincidence.
   To match a dimensional power while the normalization underneath is different
   is weaker evidence than what it looks.

What the comparison does support: the solver is numerically healthy, all the
blade sections converge, the Cp(lambda) curve and the Cp,max (around 0.45 at
lambda about 7.75) are physically reasonable for this airfoil family, and the
high wind power trend follow the reference. So it is a solid and inspectable BEM,
but it is not a closed quantitative validation, and the thrust and mid wind gaps
should be fixed before the numbers are used for real design decisions.

Design point summary (this is the model own output, not validated against the
reference):

```
rotor radius / blades        : 2.5 m / 3
lambda optimum / Cp,max      : 7.75 / 0.449
rated rotor speed            : 193 rpm
aerodynamic rated power      : about 5.0 kW
AEP (aero, Weibull c=8, k=2) : 17.9 MWh per year
```

## Some results

Cp(lambda, beta) surface, Cp and Ct against lambda, and the power and thrust
curve:

![cp surface](Outputs/Smart_Blade_Validated/cp_surface.png)
![cp ct](Outputs/Smart_Blade_Validated/cp_ct_curves.png)
![power thrust](Outputs/Smart_Blade_Validated/power_thrust_curve.png)

Spanwise loads at the design point, the S822 polar (raw plus Viterna 360), and
the blade geometry:

![sections](Outputs/Smart_Blade_Validated/sections_design_point.png)
![polar](Outputs/Smart_Blade_Validated/polar_S822.png)
![geometry](Outputs/Smart_Blade_Validated/blade_geometry.png)

## How to run it

```bash
pip install -r requirements.txt

python MAIN_simple.py                       # geometry, polars, Cp surfaces, pitch sweeps
python MAIN_performance_comp_validated.py   # full performance plus validation vs reference
```

The outputs (figures and CSVs) go to Outputs/<case_name>/. Every MAIN_ script
have a SETTINGS block marked at the top, you change there the geometry file, the
polars, the controller limits or the sweep ranges, and everything below is
generic.

## Project layout

```
BEM.py                    BEMSolver: per section solve (Picard + Ning) and rotor integration
corrections.py            BEMCorrections: tip/root loss, Glauert/Buhl high induction, induction factors
Aero.py                   polar loading, Reynolds interpolation, airfoil blending
Viterna.py                360 degree polar extrapolation
rotational_correction.py  Du Selig stall delay correction
Geometria.py              BladeGeometry: CSV blade definition plus spanwise meshing
controller.py             variable speed / pitch to feather controller
unsteady_bem.py           unsteady BEM wrapper
dynamic_inflow.py         Oye dynamic inflow model

MAIN_simple.py                      compact driver: geometry, polars, Cp surfaces
MAIN_performance.py                 full performance assessment
MAIN_performance_comp.py            performance plus high resolution Cp surface
MAIN_performance_comp_validated.py  performance plus validation against reference BEM
MAIN_modelica.py                    export for Modelica / system coupling

Inputs/                   blade geometry plus airfoil polars (smart, comercial, multi Re)
Outputs/                  generated figures and CSVs
```

## License

MIT, see LICENSE.
