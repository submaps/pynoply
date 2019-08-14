import numpy as np
from geopy.distance import vincenty
from scipy import spatial


def get_dist(lat, lon, plat, plon):
    d = vincenty((lat, lon), (plat, plon)).km
    return d


def to_Cartesian(lat, lng):
    '''
    function to convert latitude and longitude to 3D cartesian coordinates
    '''
    R = 6371  # radius of the Earth in kilometers

    x = R * np.cos(lat) * np.cos(lng)
    y = R * np.cos(lat) * np.sin(lng)
    z = R * np.sin(lat)
    return x, y, z


def deg2rad(degree):
    '''
    function to convert degree to radian
    '''
    rad = degree * 2 * np.pi / 360
    return rad


def rad2deg(rad):
    '''
    function to convert radian to degree
    '''
    degree = rad / 2 / np.pi * 360
    return degree


def distToKM(x):
    '''
    function to convert cartesian distance to real distance in km
    '''
    R = 6371  # earth radius
    gamma = 2 * np.arcsin(deg2rad(x / (2 * R)))  # compute the angle of the isosceles triangle
    dist = 2 * R * np.sin(gamma / 2)  # compute the side of the triangle
    return dist


def kmToDIST(x):
    '''
    function to convert real distance in km to cartesian distance
    '''
    R = 6371  # earth radius
    gamma = 2 * np.arcsin(x / 2. / R)

    dist = 2 * R * rad2deg(np.sin(gamma / 2.))
    return dist


def get_dtree(lats, lons):
    x, y, z = zip(*map(to_Cartesian, lats, lons))
    coordinates = list(zip(x, y, z))
    tree = spatial.cKDTree(coordinates)
    # print(tree.query(coordinates[0], k=10)
    return tree


def get_knearest(dtree, lat0, lon0, k):
    x_ref, y_ref, z_ref = to_Cartesian(lat0, lon0)
    dist, ix = dtree.query((x_ref, y_ref, z_ref), k=k)
    # lons_near = lons[ix]
    # lats_near = lats[ix]
    if isinstance(dist, list):
        dist = [distToKM(x) for x in dist]
    elif isinstance(dist, float):
        dist = distToKM(dist)
    return dist, ix


class MyDtree:
    def __init__(self, lats, lons):
        self.lats = lats
        self.lons = lons
        self.dtree = get_dtree(lats, lons)

    def get_knearest(self, lat0, lon0, k):
        return get_knearest(self.dtree, lat0, lon0, k)


def main():
    lats, lons = range(100), range(100)
    my_dtree = MyDtree(lats, lons)
    dist, ix = my_dtree.get_knearest(54, 53, 1)
    print('nearest', lats[ix], lons[ix])


if __name__ == '__main__':
    main()
