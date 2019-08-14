import numpy as np
import pandas as pd
import xarray as xr


class DataAssimilator:
    def __init__(self, points_source, ds, lat_regular, lon_regular):
        self.points_source = points_source
        self.ds = ds

