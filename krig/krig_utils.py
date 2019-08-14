import time
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pyKriging.krige import kriging
from krig.city_models import CityField, CitySlice
from krig.krig_plots import plot_city_map, plot_fix_surr_heatmap_3d_wrf


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


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f m' % (method.__name__, int(te - ts) / 60))
        return result

    return timed


@timeit
def swan_data_prep_2d_new(npoints='max'):
    f_test = pd.read_csv('./swandat/params_rmse_60.csv')
    f_test = f_test.loc[f_test['CFW'] == 0.025]
    f_test = f_test.reset_index()
    f = f_test
    if npoints != 'max': f = f_test.sample(n=npoints)
    f = f.reset_index(drop=True)
    X = f[['DRF', 'STPM']].values
    y = f['ERROR_K1'].values
    return X, y, f, f_test


@timeit
def wrf_data_prep_2d(npoints=None, pq='wrf__cws_T', sample_date='2018-08-01 00:00:00'):
    ds = xr.open_dataset('D:/heavy_files/netcat_data/wrf_cws_compare_msc_T Po Ff_gl.nc')
    ds_slice = ds.sel(date=sample_date)

    lats1d_init = ds_slice['latitude'].values
    lons1d_init = ds_slice['longitude'].values

    lats2d_init, lons2d_init = np.meshgrid(lats1d_init, lons1d_init)

    if npoints is None:
        lats = lats2d_init.flatten()
        lons = lons2d_init.flatten()
        # temperature = ds_slice[pq].values.flatten()
        target_nd = ds_slice[pq].values
        target = target_nd.flatten()
    else:
        # get random points
        idx = np.random.choice(list(range(len(lats2d_init.flatten()))), npoints)

        lats = lats2d_init.flatten()[idx]
        lons = lons2d_init.flatten()[idx]
        # temperature = ds_slice[pq].values.flatten()[: npoints]
        target_nd = ds_slice[pq].values
        target = target_nd.flatten()[idx]

    print('lats lons shapes:', lats.shape, lons.shape, target.shape)

    X = pd.DataFrame({'lat': lats, 'lons': lons}).values
    y = target
    f = target_nd  # init full field
    f_test = None
    wrf_krig_data = WrfKrigData(X, y, f, f_test, lats2d_init, lons2d_init, lats1d_init, lons1d_init, lats, lons)
    return wrf_krig_data


@timeit
def wrf_data_prep_3d(city_name, npoints=None, pq='wrf__cws_T'):
    ifile_nc_path = 'data/wrf_cws_compare_{}_T Po Ff_gl.nc'.format(city_name)
    ds = xr.open_dataset(ifile_nc_path)

    lats1d_init = ds['latitude'].values
    lons1d_init = ds['longitude'].values
    dates1d_init = list(range(len(ds['date'].values)))
    dates_nc = ds['date'].values
    dates3d, lats3d_init, lons3d_init = np.meshgrid(dates1d_init, lats1d_init, lons1d_init, indexing='ij')
    target_nd = ds[pq].values  # assume shape [dates, lats, lons]

    if npoints is None:
        lats = lats3d_init.flatten()
        lons = lons3d_init.flatten()
        dates = dates3d.flatten()
        target = target_nd.flatten()
    else:
        # get random points
        idx = np.random.choice(list(range(len(lats3d_init.flatten()))), npoints)
        lats = lats3d_init.flatten()[idx]
        lons = lons3d_init.flatten()[idx]
        dates = dates3d.flatten()[idx]
        # temperature = ds_slice[pq].values.flatten()[: npoints]
        target = target_nd.flatten()[idx]

    print('lats lons shapes:', lats.shape, lons.shape, target.shape)
    df_X = pd.DataFrame({'dates': dates, 'lat': lats, 'lons': lons})

    X = df_X.values
    y = target
    f = target_nd  # init full field
    f_test = None
    wrf_krig_data = WrfKrigData(
        X, y, f, f_test, lats3d_init, lons3d_init, lats1d_init,
        lons1d_init, lats, lons, dates1d_init, dates_nc, lats1d_init, lons1d_init
    )
    return wrf_krig_data



@timeit
def get_surf_data_2d(krig, x_krig, y_krig):
    krig_predict_vect = np.vectorize(lambda x, y: krig.predict([x, y]))
    surf = krig_predict_vect(x_krig, y_krig)
    return surf


@timeit
def get_surf_data_3d(krig, x_krig, y_krig, dates_krig):
    krig_predict_vect = np.vectorize(lambda x, y, z: krig.predict([x, y, z]))
    surf = krig_predict_vect(dates_krig, x_krig, y_krig)
    return surf

@timeit
def train_krig(X, y, optimizer='pso'):
    """
    :param X:
    :param y:
    :param optimizer: 'pso', 'ga' (much more faster)
    :return:
    """
    krig = kriging(X, y, name='multikrieg')
    krig.train(optimizer=optimizer)
    return krig

@timeit
def mesh_grid_to_netcdf(ofile_path, lats, lons, zi_pq_dict, dates):
    data_vars_dict = dict()
    for pq, pq_dates_values in zi_pq_dict.items():
        data_vars_dict[pq] = (('date', 'latitude', 'longitude'), pq_dates_values)

    ds = xr.Dataset(
        data_vars=data_vars_dict,
        coords={
            'latitude': lats[0, :] if len(lats.shape) == 2 else lats,
            'longitude': lons[:, 0] if len(lons.shape) == 2 else lons,
            'date': dates
        },
    )
    ds.set_coords(['longitude', 'latitude'])
    ds.to_netcdf(ofile_path)
    print('saved nc to:', ofile_path)


def run_wrf_3d(city_name, pq_list, need_plot=False, need_save_nc=True):
    zi_pq_dict = {}

    for pq in pq_list:
        npoints = 50
        krig_data: WrfKrigData = wrf_data_prep_3d(city_name=city_name, npoints=npoints, pq=pq)
        krig = train_krig(krig_data.X, krig_data.y, optimizer='ga')

        # lat_step, lon_step, date_step = 3, 3, 10
        lat_step, lon_step, date_step = 1, 1, 1
        if lat_step and lon_step and date_step:
            dates, lats, lons = np.meshgrid(
                krig_data.dates1d_init[::date_step],
                krig_data.lats1d_init[::lat_step],
                krig_data.lons1d_init[::lon_step],
                indexing='ij'
            )
        else:
            dates, lats, lons = np.meshgrid(
                krig_data.dates1d_init,
                krig_data.lats1d_init,
                krig_data.lons1d_init,
                indexing='ij'
            )

        print('lats lons dates shape', lats.shape, lons.shape, dates.shape)

        surf = get_surf_data_3d(krig, lats, lons, dates)
        zi_pq_dict[pq] = surf

        if need_plot:
            for sample_date in krig_data.dates1d_init[:100]:
                # plot init field
                plot_fix_surr_heatmap_3d_wrf(krig_data.f,
                                             krig_data.lats2d_init,
                                             krig_data.lons2d_init,
                                             sample_date,
                                             xlabel='lat',
                                             ylabel='lon',
                                             title='init field {} date'.format(pq))
                # plot interpolated field
                plot_fix_surr_heatmap_3d_wrf(surf,
                                             lats,
                                             lons,
                                             sample_date,
                                             xlabel='lat',
                                             ylabel='lon',
                                             title='{}, npts={} date'.format(pq, npoints))

    if need_save_nc:
        ofile_path = 'nc/{}_cws_krig.nc'.format(city_name)
        mesh_grid_to_netcdf(
            ofile_path, lats=krig_data.lats_nc, lons=krig_data.lons_nc, zi_pq_dict=zi_pq_dict, dates=krig_data.dates_nc
        )
