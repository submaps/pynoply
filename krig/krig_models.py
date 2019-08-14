class WrfKrigData:
    def __init__(self, X, y, f, f_test, lats2d_init, lons2d_init,
                 lats1d_init, lons1d_init, lats, lons,
                 dates1d_init=None, dates_nc=None, lats_nc=None, lons_nc=None):
        self.X = X
        self.y = y
        self.f = f
        self.f_test = f_test
        self.lats2d_init = lats2d_init
        self.lons2d_init = lons2d_init
        self.lats1d_init = lats1d_init
        self.lons1d_init = lons1d_init
        self.lats = lats
        self.lons = lons
        self.dates1d_init = dates1d_init
        self.dates_nc = dates_nc
        self.lats_nc = lats_nc
        self.lons_nc = lons_nc
