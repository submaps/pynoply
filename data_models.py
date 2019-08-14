import xarray as xr
import fnmatch
import re
import pandas as pd
import os

from data_utils import read_cws_resample, rename_cws


class DataObs:
    def __init__(self, obs_name: str, idir_path: str, ifile_info: str = None):
        self.obs_name = obs_name
        if ifile_info:
            self.dfs_dict = self.load_csvs(idir_path)
            self.df_info = self.load_info(ifile_info)
            self.dfs = [self.dfs_dict[_id] for _id in self.df_info['_id'].astype(str).values]
            assert len(self.dfs) == len(self.df_info)
        else:
            self.df_all = self.load_all(idir_path)
            self.dfs_dict = {_id: df_id for _id, df_id in self.df_all.groupby('_id')}
            self.dfs = list(self.dfs_dict.values())
            self.df_info = self.get_cws_df_info(self.df_all)
            assert len(self.dfs) == len(self.df_info)

    def __repr__(self):
        return self.df_info.to_string()

    def load_all(self, ifile_path):
        df = read_cws_resample(ifile_path)
        df = rename_cws(df)
        return df

    def resample_df_hour(self, df):
        df_hourly = df.groupby('_id').resample('1h').mean()
        df_hourly.reset_index(inplace=True)
        return df_hourly

    def load_csvs(self, idir_path):
        files = os.listdir(idir_path)
        df_dict = {}
        for file in files:
            _id = file.split('.')[0]
            print(_id)
            df = pd.read_csv('{}/{}'.format(idir_path, file))
            df.rename(columns={'id': '_id'}, inplace=True)
            df['_id'] = df['_id'].astype(str)
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
            df.drop_duplicates(subset=['Date'], inplace=True)
            df.dropna(subset=['Date'], inplace=True)
            df.index = df['Date'].values
            df_dict[_id] = df
        return df_dict

    def load_info(self, ifile_info):
        df_info = pd.read_csv(ifile_info)
        df_info.rename(columns={'id': '_id'}, inplace=True)
        return df_info

    def get_cws_df_info(self, df_all):
        df_info = df_all[['_id', 'lat', 'lon', 'altitude']].sort_values('_id').drop_duplicates('_id').dropna()
        return df_info


class PointsDataSource:
    def __init__(self, name, df_list, modify=True):
        self.name = name
        self.points_num = len(df_list)
        self.df_list = [self.modify_df(df, name) if modify else df for df in df_list]

    def modify_df(self, df, name):
        """
        add suffix to all columns except Date
        :param df:
        :param name:
        :return:
        """
        df = df.add_suffix('_{}'.format(name))
        df.rename(columns={'Date_{}'.format(name): 'Date'}, inplace=True)
        assert '_id_{}'.format(name) in df.columns.tolist()
        return df

    def __repr__(self):
        return 'PointsDataSource: {} {}'.format(self.name, self.points_num)

    def __add__(self, o):
        if isinstance(o, int):
            return self
        else:
            df_common_list = [pd.merge(df1, df2, on='Date', how='outer') for df1, df2 in zip(self.df_list, o.df_list)]
            print(df_common_list[0].columns.tolist())
            return PointsDataSource(name=self.name + '_' + o.name, df_list=df_common_list, modify=False)

    def __radd__(self, o):
        return self.__add__(o)


class DataEra:
    def __init__(self, expr: str, ifile_era: str):
        self.expr = expr
        ifile_nc = '{}/wrf.{}.nc'.format(ifile_era, expr)
        self.ds = self.prepare_ds_era(xr.open_dataset(ifile_nc))

    def __repr__(self):
        return 'DataEra {} {}'.format(self.expr, self.ds)

    def prepare_ds_era(self, ds):
        era_rename_dict = {
            "var134": "Po",
            "var165": "u_10m",
            "var166": "v_10m",
            "var167": "T",
        }
        ds.rename(era_rename_dict, inplace=True)
        ds['Po'] = ds['Po'] / 100
        ds['Ff'] = (ds['u_10m'] ** 2 + ds['v_10m'] ** 2) ** 0.5
        ds['T'] -= 273.15
        return ds


class DataWrf:
    def __init__(self, expr: str, idir_wrf: str):
        self.idir_wrf = idir_wrf
        nc_files = fnmatch.filter(os.listdir(idir_wrf), '*.nc')
        grid_files = [x for x in nc_files if 'monm' not in x and x != 'wrf.{}.nc'.format(expr)]
        ifile_nc = '{}/wrf.{}.nc'.format(idir_wrf, expr)
        self.ds = self.prepare_ds_wrf(xr.open_dataset(ifile_nc))
        self.grids = dict()
        for grid_file in grid_files:
            grid_name = re.findall('wrf.{}.(.*).nc'.format(expr), grid_file)[0]
            grid_file_path = '{}/wrf.{}.{}.nc'.format(idir_wrf, expr, grid_name)
            print(grid_file_path)
            self.grids.update({grid_name: self.prepare_ds_wrf(xr.open_dataset(grid_file_path))})

    def __repr__(self):
        return str(self.grids)

    def prepare_ds_wrf(self, ds_wrf):
        ds_wrf.rename(
            {'T2': 'T',
             'PSFC': 'Po'},
            inplace=True
        )
        ds_wrf['Po'] = ds_wrf['Po'] / 100
        ds_wrf['Ff'] = (ds_wrf['u_10m'] ** 2 + ds_wrf['v_10m'] ** 2) ** 0.5
        ds_wrf['T'] -= 273.15
        return ds_wrf

    def get_points_ds(self, obs_name):
        for k, ds in self.grids.items():
            if k.startswith(obs_name):  # find first
                return ds

    def get_obs_points(self, obs_name):
        ds = self.get_points_ds(obs_name)
        df_list = self.get_dfs_from_plane_ds(ds)
        return df_list

    def get_dfs_from_plane_ds(self, ds):
        df_list = []
        for i in ds.ncells.values.tolist():
            df = ds.isel(ncells=i).to_dataframe()
            df.rename(columns={'XTIME': 'Date'}, inplace=True)
            df['Date'] = df.index
            df_list.append(df)
        return df_list


class Plotter:
    def __init__(self, name_dfs_list, pq_list=('T', 'Po', 'Ff')):
        """
        suggest Date as index column
        :param name_dfs_list: [(name1 and df1_list), (name2, df2_list)...]
        """
        self.name_dfs_list = name_dfs_list
        self.names = [name for name, dfs in name_dfs_list]

    def join_all(self, source_list: list, start: str = None, end: str = None, freq: str = '1h'):
        """
        Input data:
        list of df list for each data source
        [df_cws_list, df_rp5_list, df_wrf_list, df_era_list]
        where df_list is a list of df for each point
        df_list[0] == df for point 0
        ...
        :return: df_common_list joined df for each data source
        for each point
        df_common_list[0] == df_common for point 0
        _id | date | lat | lon | pq_cws | pq_rp5 | pq_wrf | pq_era | ...
        ...
        """
        # if not start and not end:
        #     start, end = get_start_end(sum(source_list, ()))
        #
        # if end and start:
        #     proper_dates = pd.date_range(start, end, freq=freq)
        #     df_proper = pd.DataFrame({'Date': proper_dates})
        #     print('proper dates', proper_dates)
        #
        #     proper_dfs = []
        #     for name, dfs in self.name_dfs_list:
        #         dfs = [pd.merge(df_proper, df, on='Date', how='left') for df in dfs]
        # dfs2 = [pd.merge(df_proper, df, on='Date', how='left') for df in self.df2_list]
        # df_common_list = [pd.merge(df1, df2, on='Date', how='inner')  for df1, df2 in zip(dfs1, dfs2)]
        # for each point df_common
        #     return df_common_list
        # else:
        #     raise Exception('nan value found in dates')

    def __repr__(self):
        return 'Plotter: {}'.format(' '.join(self.names))
