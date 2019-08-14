import time
import numpy as np
import pandas as pd
from kd_utils import MyDtree
import xarray as xr

h7 = pd.Timedelta('7h')


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f m' % (method.__name__, int(te - ts) / 60))
        return result

    return timed


def read_cws_resample(ifile_path):
    df = pd.read_csv(ifile_path)
    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']
    return df


@timeit
def get_nearest_curvilinear_grid(ds: xr.Dataset, _id_lat_lon_list: list):
    xlong = ds.XLONG.values.flatten()
    xlat = ds.XLAT.values.flatten()
    west_east2d, south_north2d = np.meshgrid(ds.west_east, ds.south_north, indexing='ij')
    south_north = south_north2d.flatten()
    west_east = west_east2d.flatten()

    test_ds = ds.sel(south_north=33, west_east=20)
    test_lon1 = float(test_ds.XLONG.values)
    test_lon2 = float(ds.XLONG[33, 20].values)
    test_lat1 = float(test_ds.XLAT.values)
    test_lat2 = float(ds.XLAT[33, 20].values)
    assert test_lon1 == test_lon2 and test_lat1 == test_lat2
    kdtree = MyDtree(xlat, xlong)

    df_list = []
    rows_coords = []
    for _id, lat0, lon0 in _id_lat_lon_list:
        dist, ix = kdtree.get_knearest(lat0, lon0, 1)
        lat_nearest, lon_nearest = xlat[ix], xlong[ix]
        south_north_ind, west_east_ind = south_north[ix], west_east[ix]
        assert dist < 5  # dist < 5 km

        df = ds.sel(south_north=south_north_ind, west_east=west_east_ind).to_dataframe()
        df['Date'] = df.index
        df['_id'] = [_id] * len(df)
        assert _id == df['_id'].unique()[0] or _id == df['_id'].unique()[1]
        df_list.append(df)
        rows_coords.append(dict(lat0=lat0, lon0=lon0,
                                lat_nearest=lat_nearest, lon_nearest=lon_nearest,
                                south_north_ind=south_north_ind, west_east_ind=west_east_ind,
                                dist=dist))
    # pd.DataFrame(rows_coords).to_csv('coords_samples_{}.csv'.format(len(df_list)))
    # print(len(df_list))
    return df_list


def get_start_end(df_list: list):
    """
    get start end date from list of DataFrames
    :param df_list:
    :return:
    """
    start = min([df['Date'].min() for df in df_list])
    end = max([df['Date'].max() for df in df_list])
    return start, end


def rename_cws(df_cws):
    return df_cws.rename(
        columns={
            'temperature': 'T',
            'pressure': 'Po',
            'wind_anlge': 'DD',
            'wind_strength': 'Ff',
            'wind_angle': 'DD',
            'humidity': 'U',
            'date': 'Date'
        }
    )


def get_corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def autoshift_date(df, city):
    if city == 'msc':
        df.index += h7
    return df


def get_nearest_era(ds, _id_lat_lon_list_cws):
    pass
