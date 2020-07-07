# -*- coding: utf-8 -*-
"""
"IHEWAcollect_cutline.bat"

cd "D:\IHEProjects\20200218-Philippines\Code"
python IHEWAcollect_stats_EWH.py
"""
import inspect
import os
from datetime import datetime

import numpy as np
import pandas as pd
from rasterstats import zonal_stats

import gdal, ogr, osr


def Open_array_info(filename=''):
    """
    Opening a tiff info, for example size of array, projection and transform matrix.

    Keyword Arguments:
    filename -- 'C:/file/to/path/file.tif' or a gdal file (gdal.Open(filename))
        string that defines the input tiff file or gdal file

    """
    f = gdal.Open(r"%s" % filename)
    if f is None:
        print('%s does not exists' % filename)
    else:
        geo_out = f.GetGeoTransform()
        proj = f.GetProjection()
        size_X = f.RasterXSize
        size_Y = f.RasterYSize
        f = None
    return geo_out, proj, size_X, size_Y


def Save_as_tiff(name, data, geo, projection):
    """
    This function save the array as a geotiff

    Keyword arguments:
    name -- string, directory name
    data -- [array], dataset of the geotiff
    geo -- [minimum lon, pixelsize, rotation, maximum lat, rotation,
            pixelsize], (geospatial dataset)
    projection -- integer, the EPSG code
    """
    # save as a geotiff
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(name, int(data.shape[1]), int(data.shape[0]), 1,
                           gdal.GDT_Float32, ['COMPRESS=LZW'])
    srse = osr.SpatialReference()
    if projection == '':
        srse.SetWellKnownGeogCS("WGS84")

    else:
        try:
            if not srse.SetWellKnownGeogCS(projection) == 6:
                srse.SetWellKnownGeogCS(projection)
            else:
                try:
                    srse.ImportFromEPSG(int(projection))
                except:
                    srse.ImportFromWkt(projection)
        except:
            try:
                srse.ImportFromEPSG(int(projection))
            except:
                srse.ImportFromWkt(projection)

    dst_ds.SetProjection(srse.ExportToWkt())
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
    dst_ds.SetGeoTransform(geo)
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds = None
    return ()


def main(dir_shp, dir_download, dir_tmp, dir_csv,
         tif, shp, csv,
         stats_names):
    from pprint import pprint
    
    date_fmt = '%Y-%m-%d'
    
    file_shp = os.path.join(dir_shp, shp['name'])
    file_csv = os.path.join(dir_csv, csv['name'])

    if os.path.isfile(file_csv):
        os.remove(file_csv)
    with open(file_csv, 'w+', encoding='utf-8') as fp:
        # Shapefile
        ds_shp_driver = ogr.GetDriverByName('ESRI Shapefile')
        # 0 means read-only. 1 means writeable.
        ds_shp = ds_shp_driver.Open(file_shp, 0)
        if ds_shp is None:
            print('Could not open %s' % (file_shp))
            os._exit(1)
        else:
            ds_shp_layer = ds_shp.GetLayer()
            ds_shp_layer_fea_n = ds_shp_layer.GetFeatureCount()
        
        # csv column names
        col_names = []
        # stats_names = ['count', 'min', 'mean', 'max', 'median']
        for i in range(ds_shp_layer_fea_n):
            for stats_name in stats_names:
                col_names.append('{}_{}'.format(stats_name, i))
        fp.write('{dtime},{cols}\n'.format(dtime='date', cols=','.join(col_names)))
        
        # GeoTIF
        for var in tif.keys():
            dir_tif = os.path.join(dir_download, var, 'download')

            date_s = datetime.strptime(tif[var]['period']['s'], date_fmt)
            date_s_year = date_s.year
            date_s_month = date_s.month
            date_s_day = date_s.day
            date_e = datetime.strptime(tif[var]['period']['e'], date_fmt)

            date_month = date_s_month
            data = None
            ds_cols, ds_rows = np.inf, np.inf
            
            dates = pd.date_range(date_s, date_e, freq='D')
            for date in dates:
                file_tif = os.path.join(dir_tif, tif[var]['name'].format(dtime=date))
        
                if os.path.isfile(file_tif):
                    geo_trans, geo_proj, size_x, size_y = Open_array_info(file_tif)

                    ds = gdal.Open(file_tif)
                    ds_cols = int(np.min([ds_cols, ds.RasterXSize]))
                    ds_rows = int(np.min([ds_rows, ds.RasterYSize]))
                    
                    ds_band = ds.GetRasterBand(1)
                    ds_ndv = ds_band.GetNoDataValue()
                    ds_data = ds_band.ReadAsArray()
                    ds_data = np.where(ds_data == ds_ndv, 0.0, ds_data)

                    date_year = date.year
                    date_day = date.day
                    
                    # calculate
                    date_str = '{dtime:%Y-%m-%d}'.format(dtime=date)
                    # print(date_str, 'zonal_stats')
                    
                    ds_stats = zonal_stats(file_shp, file_tif, stats=' '.join(stats_names))
                    fp.write(date_str)
                    for i in range(ds_shp_layer_fea_n):
                        for key in stats_names:
                            fp.write(',{}'.format(ds_stats[i][key]))
                    fp.write('\n')
                    
                    if date.is_month_start:
                        file_tif_monthly = os.path.join(dir_tmp, tif[var]['output'].format(dtime=date))
                        print(file_tif_monthly)
                        
                        data = np.where(data == 0.0, ds_ndv, data)
                        Save_as_tiff(name=file_tif_monthly, data=ds_data, geo=geo_trans, projection="WGS84")


                    # print('{:02d} {} {} {} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(
                    #           date_day, [ds_rows, ds_cols], ds_data.shape, ds_ndv,
                    #           np.mean(data), np.nanmax(data), np.nanmean(data),
                    #           np.mean(ds_data), np.nanmax(ds_data), np.nanmean(ds_data)))
                
                # if date_year > 2005:
                #     os._exit(0)


if __name__ == "__main__":
    # path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(
        os.getcwd(),
        os.path.dirname(
            inspect.getfile(
                inspect.currentframe())),
        '../'
    )
    
    dir_shp = os.path.join(path, 'Data', 'Shapefile')
    dir_download = os.path.join(path, 'Data', 'IHEWAcollect')
    
    dir_tmp = os.path.join(path, 'Data', 'Output', 'tmp')
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)

    dir_csv = os.path.join(path, 'Data', 'Output', 'csv')
    if not os.path.exists(dir_csv):
        os.makedirs(dir_csv)
    
    stats_names = ['count', 'min', 'mean', 'max', 'median']
    shp = {
        'name': 'Mindanao-RiverBasin.shp'
    }

    date_s = '2013-12-01'
    date_e = '2020-01-01'

    products = {
        'EWH-CSR-v3.1':{
            'tif': {
                'EWH': {
                    'name': 'CSR_v3.1_mm_D-{dtime:%Y%m%d}.tif',
                    'version': 'v3.1',
                    'resolution': 'daily',
                    'variable': 'EWH',
                    'period': {
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'CSR_v3.1_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'EWH-CSR-v3.1.csv'
            }
        },
        'EWH-CSR-v3.2':{
            'tif': {
                'EWH': {
                    'name': 'CSR_v3.2_mm_D-{dtime:%Y%m%d}.tif',
                    'version': 'v3.2',
                    'resolution': 'daily',
                    'variable': 'EWH',
                    'period': {
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'CSR_v3.2_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'EWH-CSR-v3.2.csv'
            }
        },
        'EWH-GFZ-v3.1':{
            'tif': {
                'EWH': {
                    'name': 'GFZ_v3.1_mm_D-{dtime:%Y%m%d}.tif',
                    'version': 'v3.1',
                    'resolution': 'daily',
                    'variable': 'EWH',
                    'period': {
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'GFZ_v3.1_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'EWH-GFZ-v3.1.csv'
            }
        },
        'EWH-GFZ-v3.2':{
            'tif': {
                'EWH': {
                    'name': 'GFZ_v3.2_mm_D-{dtime:%Y%m%d}.tif',
                    'version': 'v3.2',
                    'resolution': 'daily',
                    'variable': 'EWH',
                    'period': {
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'GFZ_v3.2_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'EWH-GFZ-v3.2.csv'
            }
        },
        'EWH-JPL-v3.1':{
            'tif': {
                'EWH': {
                    'name': 'JPL_v3.1_mm_D-{dtime:%Y%m%d}.tif',
                    'version': 'v3.1',
                    'resolution': 'daily',
                    'variable': 'EWH',
                    'period': {
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'JPL_v3.1_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'EWH-JPL-v3.1.csv'
            }
        },
        'EWH-JPL-v3.2':{
            'tif': {
                'EWH': {
                    'name': 'JPL_v3.2_mm_D-{dtime:%Y%m%d}.tif',
                    'version': 'v3.2',
                    'resolution': 'daily',
                    'variable': 'EWH',
                    'period': {
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'JPL_v3.2_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'EWH-JPL-v3.2.csv'
            }
        },
    }
    
    for prod_key, prod_val in products.items():
        # print(prod_key, prod_val['tif'])
        main(dir_shp, dir_download, dir_tmp, dir_csv,
             prod_val['tif'], shp, prod_val['csv'],
             stats_names)
