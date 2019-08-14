import matplotlib.pyplot as plt

from data_models import PointsDataSource, DataWrf, DataObs
from data_utils import get_nearest_curvilinear_grid
from stats_models import Stats


def stats_exp(obs_data_cws, obs_data_rp5, wrf_data, _id_lat_lon_list_cws,
              _id_lat_lon_list_rp5, ofile_stats_cws_path, ofile_stats_rp5_path):
    df_stats_cws = Stats(('cws', obs_data_cws.dfs),
                         ('wrf', get_nearest_curvilinear_grid(wrf_data.ds, _id_lat_lon_list_cws))) \
        .calc_stats()
    df_stats_cws. \
        round(2) \
        .to_csv(ofile_stats_cws_path, index=False, sep='\t', decimal=',')
    df_stats_cws.groupby('pq') \
        .mean() \
        .round(2) \
        .to_csv(ofile_stats_cws_path.replace('.csv', '.mean.csv'), sep='\t', decimal=',')

    df_stats_rp5 = Stats(('rp5', obs_data_rp5.dfs),
                         ('wrf', get_nearest_curvilinear_grid(wrf_data.ds, _id_lat_lon_list_rp5))) \
        .calc_stats()
    df_stats_rp5 \
        .round(2) \
        .to_csv(ofile_stats_rp5_path, index=False, sep='\t', decimal=',')
    df_stats_rp5.groupby('pq') \
        .mean() \
        .round(2) \
        .to_csv(ofile_stats_rp5_path.replace('.csv', '.mean.csv'), sep='\t', decimal=',')


def plot_exp(source_list, city_name, pq_list=('T', 'Po', 'Ff')):
    point_data_source_list = []
    for name, df_list in source_list:
        data_source = PointsDataSource(name=name, df_list=df_list)
        point_data_source_list.append(data_source)
    merged_data_source = sum(point_data_source_list)  # merge dfs from both sources
    df_common_list = merged_data_source.df_list
    print(df_common_list[0].columns.tolist())
    print(df_common_list[0].head().to_string())

    for i, df in enumerate(df_common_list):
        if len(df) > 0:
            id_col = [c for c in df.columns if c.startswith('_id_')][0]
            for pq in pq_list:
                title = '{} {} station {}  {}'.format(city_name, pq, i, df[id_col].values[0])
                ofile_path = 'img/timeseries/{}.png'.format(title).replace(':', '_')
                plot_point_data(df, pq, title, ofile_path)


def plot_point_data(df, pq, title, ofile_path=None):
    plt.figure(figsize=(20, 10))
    plt.title(title)
    pq_cols = [col for col in df.columns.tolist() if col.startswith('{}_'.format(pq))]
    for pq_col in pq_cols:
        plt.plot(df['Date'].values, df[pq_col].values, label=pq_col)
    plt.legend()
    plt.xticks(rotation=45)
    if ofile_path:
        plt.savefig(ofile_path)
    else:
        plt.show()
    plt.close()


def load_datasets(city_name):
    if city_name == 'spb':
        idir_nc = r'D:\wrf_runner_data\wrfout_final_spb_127_hr_new'
        idir_obs = r'C:\HOME\netcat\data\city_obs\spb_obs'
        ifile_obs_info = r'C:\HOME\netcat\data\city_obs\rp5_city_stations_spb.csv'
        ifile_cws = r'C:\HOME\netcat\cws_obs\spb_resampled_all.csv'
        expr = 'spb_127_hr'
    else:
        idir_nc = r'D:\wrf_runner_data\wrfout_final_msc_126_hr_prev'
        idir_obs = r'C:\HOME\netcat\data\city_obs\msc_obs'
        ifile_obs_info = r'C:\HOME\netcat\data\city_obs\rp5_city_stations_msc.csv'
        ifile_cws = r'C:\HOME\netcat\cws_obs\msc_resampled_all.csv'
        expr = 'msc_126_hr'

    ofile_stats_cws_path = r'C:\HOME\netcat\data\city_obs\stats\{}_cws_stats.csv'.format(expr)
    ofile_stats_rp5_path = r'C:\HOME\netcat\data\city_obs\stats\{}_rp5_stats.csv'.format(expr)

    obs_data_rp5 = DataObs('rp5', idir_obs, ifile_obs_info)
    obs_data_cws = DataObs('cws', ifile_cws)
    return  expr, idir_nc, obs_data_rp5, obs_data_cws


def main():
    # city_name = 'spb'
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

    _id_lat_lon_list_cws = obs_data_cws.df_info[['_id', 'lat', 'lon']].values.tolist()
    _id_lat_lon_list_rp5 = obs_data_rp5.df_info[['_id', 'lat', 'lon']].values.tolist()

    # stats_exp(obs_data_cws, obs_data_rp5, wrf_data, _id_lat_lon_list_cws,
    #           _id_lat_lon_list_rp5, ofile_stats_cws_path, ofile_stats_rp5_path)

    source_list = [('cws', obs_data_cws.dfs),
                   ('wrf', get_nearest_curvilinear_grid(wrf_data.ds, _id_lat_lon_list_cws)),
                   # ('era', get_nearest_era(era_data.ds, _id_lat_lon_list_cws))
                   ]
    plot_exp(source_list, city_name)

    source_list = [('rp5', obs_data_rp5.dfs),
                   ('wrf', get_nearest_curvilinear_grid(wrf_data.ds, _id_lat_lon_list_rp5)),
                   # ('era', get_nearest_era(era_data.ds, _id_lat_lon_list_cws))
                   ]
    plot_exp(source_list, city_name)


if __name__ == '__main__':
    main()
