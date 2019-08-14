import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.basemap import Basemap

from data_utils import timeit
from krig.city_models import CitySlice, CityField


def plot_city_map(city_slice: CitySlice, field: CityField, need_show=True):
    plt.figure(figsize=(10, 10))
    if city_slice.city_name == 'msc':
        lat0, lon0 = 55.75378, 37.6223
    elif city_slice.city_name == 'spb':
        lat0, lon0 = 59.92697, 30.31852

    dx = 0.3
    lon_min = lon0 - dx
    lon_max = lon0 + dx
    lat_min = lat0 - dx
    lat_max = lat0 + dx

    print('map borders:', lon_min, lat_min, lon_max, lat_max)
    bm = Basemap(llcrnrlon=lon_min, urcrnrlat=lat_max,
                 urcrnrlon=lon_max, llcrnrlat=lat_min,
                 resolution='h', epsg=4326)
    bm.drawcoastlines()

    # bm.bluemarble(alpha=0.3)
    bmlons, bmlats = bm(city_slice.points_lons, city_slice.points_lats)
    bmxi, bmyi = bm(field.xi, field.yi)
    gauss_field = gaussian_filter(field.zi, 1.)
    contourf = bm.contourf(bmyi, bmxi, gauss_field, alpha=0.6, cmap='rainbow')
    contour = bm.contour(bmyi, bmxi, gauss_field, colors='k', interpolation='none')

    serverurl = 'http://vmap0.tiles.osgeo.org/wms/vmap0?'
    bm.wmsimage(
        serverurl, xpixels=900, verbose=True,
        layers=['basic', 'river', 'inwater', 'ctylabel', 'rail', 'priroad', 'secroad'],
        format='jpeg'
    )
    bm_scatter_plot = bm.scatter(bmlons, bmlats, s=10, color='red')
    plt.colorbar(contourf)
    expr_name = '{} {}'.format(city_slice.expr, city_slice.date)
    plt.title(expr_name)
    img_ofile = 'img/{}_{}.png'.format(expr_name, field.xi.shape[0]).replace(':', '_')
    print(img_ofile)
    plt.savefig(img_ofile, dpi=300)

    if need_show:
        plt.show()


@timeit
def plot_fix_surr_heatmap_3d_wrf(surf, lats2d, lons2d, sample_date, xlabel, ylabel, title, need_show=False):
    if len(lats2d.shape) == 3:
        data = surf[sample_date]  # get date
        print('fix lats2d', sample_date)
        lats2d = lats2d[sample_date]
        lons2d = lons2d[sample_date]
    else:
        data = surf
    plt.rcParams.update({'font.size': 48})
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    color_mesh = ax.pcolormesh(lats2d, lons2d, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.contour(lats2d, lons2d, gaussian_filter(data, 1.), 10, colors='k', interpolation='none')
    plt.colorbar(color_mesh)
    plt.title(title)
    ofile_png = 'krig_n_{}_3d.png'.format(title)
    plt.savefig(ofile_png)
    if need_show:
        plt.show()
    plt.close()
    print('saved ', ofile_png)


def plot_fix_surr_heatmap_2d_wrf_city(
        surf, lats2d, lons2d, points_lats, points_lons,
        title=None, city_name=None, date=None, pq=None):

    city_slice = CitySlice(
                 city_name=city_name, date=date, df_slice=None,
                 points_lons=points_lons, points_lats=points_lats, pq=pq, expr=title
    )
    city_field = CityField(lats2d, lons2d, surf)
    plot_city_map(city_slice, city_field, need_show=True)


def plot_fix_surr_heatmap_2d_wrf(surf, lats2d, lons2d, latlabel, lonlabel,
                                 title=None):
    plt.rcParams.update({'font.size': 48})
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    color_mesh = ax.pcolormesh(lons2d, lats2d, surf.T)
    plt.xlabel(lonlabel)
    plt.ylabel(latlabel)
    plt.contour(lons2d, lats2d, gaussian_filter(surf.T, 1.), 10, colors='k', interpolation='none')
    plt.colorbar(color_mesh)
    plt.title(title)
    ofile_png = 'krig_n_{}.png'.format(title)
    plt.savefig(ofile_png)
    plt.show()
    plt.close()
    print('saved ', ofile_png)
