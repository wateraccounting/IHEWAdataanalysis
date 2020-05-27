# -*- coding: utf-8 -*-
"""
"IHEWAcollect_cutline.bat"

cd "D:\IHEProjects\20200218-Philippines\Code"
python IHEWAcollect_stats.py
"""
import inspect
import os
import shutil
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
                
                if date_month == date.month:
                    date_out = pd.Timestamp('{:04d}-{:02d}-{}'.format(date_year, date_month, '01'))
                    date_str = '{dtime:%Y-%m}'.format(dtime=date_out)
                    # print(date.strftime('%Y-%m-%d'))
                    
                    if data is None:
                        data = np.zeros((ds_rows, ds_cols))
                    data[0:ds_rows, 0:ds_cols] += ds_data[0:ds_rows, 0:ds_cols]
                    
                    if date == date_e:
                        # print(date_str, 'zonal_stats')
                        
                        # if tif[var]['resolution'] == 'monthly':
                        if tif[var]['name'].format(dtime=date_out).split('_')[2] == 'mm.m' or \
                            tif[var]['name'].format(dtime=date_out).split('_')[2] == 'mm':
                            file_tif_monthly = os.path.join(dir_tmp, tif[var]['output'].format(dtime=date_out))
                            print(file_tif_monthly)

                            shutil.copyfile(file_tif, file_tif_monthly)
                        
                            ds_stats = zonal_stats(file_shp, file_tif, stats=' '.join(stats_names))
                            fp.write(date_str)
                            for i in range(ds_shp_layer_fea_n):
                                for key in stats_names:
                                    fp.write(',{}'.format(ds_stats[i][key]))
                            fp.write('\n')
                        else:
                            file_tif_monthly = os.path.join(dir_tmp, tif[var]['output'].format(dtime=date_out))
                            print(file_tif_monthly)

                            data = np.where(data == 0.0, ds_ndv, data)
                            Save_as_tiff(name=file_tif_monthly, data=data, geo=geo_trans, projection="WGS84")

                            ds_stats = zonal_stats(file_shp, file_tif_monthly, stats=' '.join(stats_names))
                            fp.write(date_str)
                            for i in range(ds_shp_layer_fea_n):
                                for key in stats_names:
                                    fp.write(',{}'.format(ds_stats[i][key]))
                            fp.write('\n')
                else:
                    if date.month == 1:
                        date_year += -1
                    date_out = pd.Timestamp('{:04d}-{:02d}-{}'.format(date_year, date_month, '01'))
                    date_str = '{dtime:%Y-%m}'.format(dtime=date_out)
                    # print(date_str, 'zonal_stats')
                    
                    # if tif[var]['resolution'] == 'monthly':
                    if tif[var]['name'].format(dtime=date_out).split('_')[2] == 'mm.m' or \
                        tif[var]['name'].format(dtime=date_out).split('_')[2] == 'mm':
                        file_tif_monthly = os.path.join(dir_tmp, tif[var]['output'].format(dtime=date_out))
                        print(file_tif_monthly)

                        shutil.copyfile(file_tif, file_tif_monthly)
                    
                        ds_stats = zonal_stats(file_shp, file_tif, stats=' '.join(stats_names))
                        fp.write(date_str)
                        for i in range(ds_shp_layer_fea_n):
                            for key in stats_names:
                                fp.write(',{}'.format(ds_stats[i][key]))
                        fp.write('\n')
                    else:
                        file_tif_monthly = os.path.join(dir_tmp, tif[var]['output'].format(dtime=date_out))
                        print(file_tif_monthly)

                        data = np.where(data == 0.0, ds_ndv, data)
                        Save_as_tiff(name=file_tif_monthly, data=data, geo=geo_trans, projection="WGS84")

                        ds_stats = zonal_stats(file_shp, file_tif_monthly, stats=' '.join(stats_names))
                        fp.write(date_str)
                        for i in range(ds_shp_layer_fea_n):
                            for key in stats_names:
                                fp.write(',{}'.format(ds_stats[i][key]))
                        fp.write('\n')
                        
                    date_month = date.month
                    data = ds_data
                        
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

    products = {
        'ETA-1':{
            'tif': {
                'ETA': {
                    'name': 'ALEXI_v1_mm.d_D-{dtime:%Y%m%d}.tif',
                    'version': 'v1',
                    'resolution': 'daily',
                    'variable': 'ETA',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'ALEXI_v1_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'ETA-ALEXI.csv'
            }
        },
        'ETA-2':{
            'tif': {
                'ETA': {
                    'name': 'CMRSET_v1_mm.m_MS-{dtime:%Y%m}.tif',
                    'version': 'v1',
                    'resolution': 'monthly',
                    'variable': 'ETA',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'CMRSET_v1_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'ETA-CMRSET.csv'
            }
        },
        'ETA-3':{
            'tif': {
                'ETA': {
                    'name': 'GLDAS_v2.1_mm.d_MS-{dtime:%Y%m}.tif',
                    'version': 'v2.1',
                    'resolution': 'monthly',
                    'variable': 'ETA',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'GLDAS_v2.1_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'ETA-GLDAS.csv'
            }
        },
        'ETA-4':{
            'tif': {
                'ETA': {
                    'name': 'GLEAM_v3.3b_mm.m_MS-{dtime:%Y%m}.tif',
                    'version': 'v3.3b',
                    'resolution': 'monthly',
                    'variable': 'ETA',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'GLEAM_v3.3b_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'ETA-GLEAM.csv'
            }
        },
        'ETA-5':{
            'tif': {
                'ETA': {
                    'name': 'MOD16A2_v6_mm.d_D-{dtime:%Y%m%d}.tif',
                    'version': 'v6',
                    'resolution': 'eight_daily',
                    'variable': 'ETA',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'MOD16A2_v6_mm.m_MS-{dtime:%Y%m}.tif'
                }
            },
            'csv': {
                'name': 'ETA-MOD16A2.csv'
            }
        },
        'ETA-6':{
            'tif': {
                'ETA': {
                    'name': 'SSEBop_v4_mm_MS-{dtime:%Y%m}.tif',
                    'version': 'v4',
                    'resolution': 'monthly',
                    'variable': 'ETA',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'SSEBop_v4_mm.m_MS-{dtime:%Y%m}.tif'
                }
            },
            'csv': {
                'name': 'ETA-SSEBop.csv'
            }
        },
        'PCP-1':{
            'tif': {
                'PCP': {
                    'name': 'CHIRPS_v2.0_mm.m_MS-{dtime:%Y%m}.tif',
                    'version': 'v2.0',
                    'resolution': 'monthly',
                    'variable': 'PCP',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'CHIRPS_v2.0_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'PCP-CHIRPS.csv'
            }
        },
        'PCP-2':{
            'tif': {
                'PCP': {
                    'name': 'GPM_v6_mm.d_MS-{dtime:%Y%m}.tif',
                    'version': 'v6',
                    'resolution': 'monthly',
                    'variable': 'PCP',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'GPM_v6_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'PCP-GPM.csv'
            }
        },
        'PCP-3':{
            'tif': {
                'PCP': {
                    'name': 'TRMM_v7_mm.d_MS-{dtime:%Y%m}.tif',
                    'version': 'v6',
                    'resolution': 'monthly',
                    'variable': 'PCP',
                    'period': {
                        's': '2005-01-01',
                        'e': '2012-12-31'
                    },
                    'output': 'TRMM_v7_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'PCP-TRMM.csv'
            }
        },
    }
    
    # products = {
    # }
    for prod_key, prod_val in products.items():
        # print(prod_key, prod_val['tif'])
        main(dir_shp, dir_download, dir_tmp, dir_csv,
             prod_val['tif'], shp, prod_val['csv'],
             stats_names)
            
