import geopandas as gpd
import numpy as np
from shapely.geometry import Point
#from osgeo import gdal
#gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')
coastline_data = gpd.read_file('coastline.shp')
coastline = gpd.GeoSeries(coastline_data.geometry.union_all())

from math import cos, sin, asin, sqrt, radians
def calc_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]) # convert decimal degrees to radians
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

def calc_distance_to_coastline(longitude,latitude ):
    target_coordinate=Point(longitude,latitude )
    return coastline.distance(target_coordinate).values[0]

def distance_degrees_to_kilometers(distance,coord=[0,0]):
    coord_plus=[c+distance for c in coord]
    coord_minus=[c-distance for c in coord]
    return (calc_distance(*coord,*coord_plus)+calc_distance(*coord,*coord_minus))*0.5

def calc_distance_to_coastline_km(longitude,latitude ):
    target_coordinate=Point(longitude,latitude )
    return distance_degrees_to_kilometers(coastline.distance(target_coordinate).values[0],[longitude,latitude])