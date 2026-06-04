
import numpy as np
import pandas as pd

class BladeGeometry:
    def __init__(
        self,
        file_path=None,
        n_sections=30,
        B=3,
        hub_diameter=0.2,
        r_is_span_offset=True,
    ):
        self.file_path = file_path
        self.n_sections = n_sections

        self.B = B
        self.hub_diameter = hub_diameter
        self.r_hub = hub_diameter / 2
        self.r_is_span_offset = r_is_span_offset

        self.r = None
        self.chord = None
        self.twist = None
        self.airfoil = None

        self.R = None

    def load_csv(self):
        df = pd.read_csv(self.file_path)
        df = df.sort_values(by="r")

        self.span = df["r"].values           # span offset from hub centre [m]
        self.chord = df["chord"].values      # [m]
        self.twist = np.radians(df["twist"]) # twist a radianes
        self.airfoil = df["airfoil"].values  # airfoil local

        # Default behavior preserves the existing convention:
        # CSV 'r' is treated as span offset from hub centre.
        # Set r_is_span_offset=False only if the CSV already stores absolute radius.
        assert self.span[0] >= 0, (
            "CSV 'r' column must be a non-negative span offset from the hub centre."
        )

        if self.r_is_span_offset:
            self.r = self.r_hub + self.span
        else:
            self.r = self.span.copy()
        self.R = self.r[-1]
        
        self.airfoil_r = self.r.copy()
        self.airfoil_names = self.airfoil.copy()

    def Malla(self):
        r_min = self.r.min()
        r_max = self.r.max()
    
        r_edges = np.linspace(r_min, r_max, self.n_sections + 1)
        r_new = 0.5 * (r_edges[:-1] + r_edges[1:])
    
        chord_new = np.interp(r_new, self.r, self.chord)
        twist_new = np.interp(r_new, self.r, self.twist)
    
        self.r = r_new
        self.chord = chord_new
        self.twist = twist_new
        # self.airfoil is left alone on purpose, not regridded.
        # use self.airfoil_r and self.airfoil_names for the blending lookups


    def get_section(self, i):
        return {
            "r": self.r[i],
            "chord": self.chord[i],
            "twist": self.twist[i]
        }

    def n(self):
        return len(self.r)
