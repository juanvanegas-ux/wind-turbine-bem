"""Generate Jorge's real +/-180 deg multi-Reynolds polar CSVs into
Inputs/jorge_polars/ so the validated MAIN variant is self-contained.

Reads the raw column tables ({af}_Alpha/_Cl/_Cd/_Re.txt) that Jorge shipped
and writes one CSV per (airfoil, Re) with columns alpha,cl,cd. These are the
genuine tabulated polars (no Viterna extrapolation needed -- they already span
-180..+180 deg), which is what makes the closure match ~4%.
"""
import os
import numpy as np
import pandas as pd

JT = r"C:\Users\vaneg\Desktop\Code\Codigo\Codigo\BEM"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Inputs", "jorge_polars")


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for af in ("cylinder", "S822", "S823"):
        al = np.loadtxt(os.path.join(JT, f"{af}_Alpha.txt"))
        cl = np.loadtxt(os.path.join(JT, f"{af}_Cl.txt"))
        cd = np.loadtxt(os.path.join(JT, f"{af}_Cd.txt"))
        re = np.atleast_1d(np.loadtxt(os.path.join(JT, f"{af}_Re.txt")))
        if cl.ndim == 1:
            cl = cl[:, None]
            cd = cd[:, None]
        entry = {}
        for j, rev in enumerate(re):
            fname = f"{af}_Re{int(rev)}.csv"
            path = os.path.join(OUT, fname)
            pd.DataFrame({"alpha": al, "cl": cl[:, j],
                          "cd": cd[:, j]}).to_csv(path, index=False)
            entry[int(rev)] = fname
        manifest[af] = entry
        print(f"{af}: {len(re)} Re tables, alpha {al.min():.0f}..{al.max():.0f}")
    print(f"\nwrote to {OUT}")
    for af, e in manifest.items():
        print(f"  {af}: " + ", ".join(str(r) for r in e))


if __name__ == "__main__":
    main()
