import numpy as np
import pandas as pd
import xarray as xr

from data_models import DataWrf
from data_utils import get_nearest_curvilinear_grid, timeit
from pynoply_stats import load_datasets


class DataAssimilator:
    def __init__(self, method, ds, pq):
        self.method = method
        self.ds = ds
        self.pq = pq

    def fit(self, points_source):
        self.df_source_list = get_nearest_curvilinear_grid(self.ds, points_source)
        self.df_train = pd.concat(self.df_source_list, ignore_index=True)[['Date', 'lat', 'lon', self.pq]]
        print('df_source_list len:', len(self.df_train))

    def transform(self, predict_points_source):
        # to do
        pass

    def __repr__(self):
        return 'DataAssimilator method {}; dims: {}; vars: {}'.format(
            self.method,
            ' '.join(list(self.ds.dims)),
            ' '.join(list(self.ds.variables))
        )

@timeit
def main():
    city_name = 'msc'
    expr, idir_nc, obs_data_rp5, obs_data_cws = load_datasets(city_name)
    wrf_data = DataWrf(expr, idir_nc)
    # era_data = DataEra(exp, idir_era_nc)
    print(wrf_data)
    print()
    print(obs_data_rp5)

    df_wrf_rp5_list = wrf_data.get_obs_points('rp5')
    df_wrf_cws_list = wrf_data.get_obs_points('cws')

    print('wrf rp5 list:', len(df_wrf_rp5_list), len(obs_data_rp5.dfs))
    print('wrf cws list:', len(df_wrf_cws_list), len(obs_data_cws.dfs))

    method = 'krig'
    n = 100
    xlong_flatten = wrf_data.ds.XLONG.values.flatten()
    xlat_flatten = wrf_data.ds.XLAT.values.flatten()
    dates_flatten = wrf_data.ds.XTIME.values.flatten()

    print('xlong xlat dates shapes:', xlong_flatten.shape, xlat_flatten.shape, dates_flatten.shape)
    lats_regular = np.linspace(xlat_flatten.min(), xlat_flatten.max(), n)
    lons_regular = np.linspace(xlong_flatten.min(), xlong_flatten.max(), n)
    dates_regular = dates_flatten
    da = DataAssimilator(method, wrf_data.ds)
    print(da)
    _id_lat_lon_list_cws = obs_data_cws.df_info[['_id', 'lat', 'lon']].values.tolist()
    _id_lat_lon_list_rp5 = obs_data_rp5.df_info[['_id', 'lat', 'lon']].values.tolist()
    da.fit(_id_lat_lon_list_rp5)
    ds_res = da.transform(df_wrf_cws_list)
    print(ds_res)


if __name__ == '__main__':
    main()
