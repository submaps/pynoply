class CityField:
    def __init__(self, xi, yi, zi):
        self.xi = xi
        self.yi = yi
        self.zi = zi


class CityGrid:
    def __init__(self, xi, yi, coords, xi_1d_init, yi_1d_init):
        self.xi = xi
        self.yi = yi
        self.coords = coords
        self.xi_1d_init = xi_1d_init
        self.yi_1d_init = yi_1d_init


class CitySlice:
    def __init__(self, city_name, date, df_slice,
                 points_lons, points_lats, pq, expr
                 ):
        self.date = date
        self.df_slice = df_slice
        self._id_list = df_slice['_id'].unique() if df_slice is not None else None
        self.points_lons = points_lons
        self.points_lats = points_lats
        # self.x = x
        # self.y = y
        # self.field = field
        # self.grid_coords = grid.coords
        self.expr = expr
        self.pq = pq
        self.city_name = city_name
