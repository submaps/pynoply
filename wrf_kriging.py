import numpy as np

from krig.krig_plots import plot_fix_surr_heatmap_2d_wrf_city, plot_fix_surr_heatmap_3d_wrf
from krig.krig_utils import wrf_data_prep_2d, train_krig, get_surf_data_2d, WrfKrigData, wrf_data_prep_3d, \
    get_surf_data_3d, mesh_grid_to_netcdf, timeit


def run_wrf_2d(pq, city_name):
    npoints = 150
    sample_dates = ['2018-08-01 00:00:00', '2018-08-01 01:00:00', '2018-08-01 02:00:00', '2018-08-01 03:00:00']

    for sample_date in sample_dates:
        krig_data = wrf_data_prep_2d(npoints=npoints, pq=pq, sample_date=sample_date)
        krig = train_krig(krig_data.X, krig_data.y, optimizer='ga')

        lat_step, lon_step = 3, 3
        if lat_step and lon_step:
            init_lats, init_lons = krig_data.lats1d_init[::lat_step], krig_data.lons1d_init[::lon_step]
        else:
            init_lats, init_lons = krig_data.lats1d_init, krig_data.lons1d_init

        lats, lons = np.meshgrid(init_lats, init_lons)
        print('lats lons shape', lats.shape, lons.shape)
        surf = get_surf_data_2d(krig, lats, lons)

        plot_fix_surr_heatmap_2d_wrf_city(krig_data.f,
                                     krig_data.lats2d_init,
                                     krig_data.lons2d_init,
                                     krig_data.lats,
                                     krig_data.lons,
                                     title='init field {}'.format(pq),
                                     city_name=city_name,
                                     date=sample_date.replace(':', '_'),
                                     pq=pq)

        plot_fix_surr_heatmap_2d_wrf_city(surf,
                                         lats,
                                         lons,
                                         krig_data.lats,
                                         krig_data.lons,
                                         title='{}, npts={}'.format(pq, npoints),
                                         city_name=city_name,
                                         date=sample_date.replace(':', '_'),
                                         pq=pq)


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
                plot_fix_surr_heatmap_3d_wrf(krig_data.f,
                                             krig_data.lats2d_init,
                                             krig_data.lons2d_init,
                                             sample_date,
                                             xlabel='lat',
                                             ylabel='lon',
                                             title='init field {} date'.format(pq))

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


@timeit
def main():
    city_name = 'msc'
    need_cols = [
                 'T',
                 'Po',
                 'Ff',
                 'wrf__cws_T',
                 'wrf__cws_Po',
                 'wrf__cws_Ff'
    ]
    run_wrf_3d(city_name, need_cols)  # use time coord

    # for pq_name in need_cols:
    #     run_wrf_2d(pq_name, city_name)


if __name__ == '__main__':
    main()

# (50,) (50,) (50,)
# 'wrf_data_prep_2d_new'  0.00 m
# 'train_krig'  0.47 m
# 'get_surf_data_new'  36.22 m
# 'main'  36.75 m


# D:\Anaconda3\python.exe C:/HOME/krig/wrf_kriging.py
# lats lons shapes: (50,) (50,) (50,)
# 'wrf_data_prep_3d'  0.00 m
# 'train_krig'  0.10 m
# lats lons dates shape (1342, 75, 75) (1342, 75, 75) (1342, 75, 75)
# 'get_surf_data_3d'  84.82 m
# lats lons shapes: (50,) (50,) (50,)
# 'wrf_data_prep_3d'  0.00 m
# 'train_krig'  0.10 m
# lats lons dates shape (1342, 75, 75) (1342, 75, 75) (1342, 75, 75)
# 'get_surf_data_3d'  84.40 m
# saved nc to: nc/msc_cws_krig.nc
# 'mesh_grid_to_netcdf'  0.00 m
# 'main'  169.47 m
