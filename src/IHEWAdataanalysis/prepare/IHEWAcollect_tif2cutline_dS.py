"""
cd "D:\IHEProjects\20200218-Philippines\Code"
python IHEWAcollect_tif2cutline_dS.py


set in_path="D:\IHEProjects\20200218-Philippines\Data\Output\tmp"
set out_path="D:\IHEProjects\20200218-Philippines\Data\Output\tif"

set shapefile_path="D:\IHEProjects\20200218-Philippines\Data\Shapefile\Mindanao-RiverBasin.shp"


if exist %out_path% (echo yes) else (echo no && mkdir %out_path%)

FORFILES /p %in_path% /s ^
/m *.tif ^
/C "cmd /Q /c for %%I in (@file) do gdalwarp -of GTiff -overwrite -s_srs epsg:4326-t_srs epsg:4326 -r near -tr 0.05 0.05 -cutline %shapefile_path% -crop_to_cutline %in_path%\%%~I %out_path%\%%~I"

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


def main(dir_in, dir_out, tif, file_shp, cmd1,cmd2):
    date_fmt = '%Y-%m-%d'
    
    for var in tif.keys():
        date_s = datetime.strptime(tif[var]['period']['s'], date_fmt)
        date_s_year = date_s.year
        date_s_month = date_s.month
        date_s_day = date_s.day
        date_e = datetime.strptime(tif[var]['period']['e'], date_fmt)
        
        date_year = date_s_year
        data = None
        ds_cols, ds_rows = np.inf, np.inf

        dates = pd.date_range(date_s, date_e, freq='MS')
        
        nmonth = 0.0
        # for idate in range(1, len(dates) - 1):
        for idate in range(len(dates)):
            date_o = dates[idate]
            date_l = dates[idate] - pd.tseries.offsets.DateOffset(months=1)
            date_r = dates[idate] + pd.tseries.offsets.DateOffset(months=1)

            fname_o = tif[var]['output'].format(dtime=date_o)
            fname_l = tif[var]['output'].format(dtime=date_l)
            fname_r = tif[var]['output'].format(dtime=date_r)

            file_i_l = os.path.join(dir_in, fname_l)
            file_i_r = os.path.join(dir_in, fname_r)
            file_o = os.path.join(dir_out, fname_o)
            
            if os.path.isfile(file_i_l) and os.path.isfile(file_i_r):
                nmonth += 1.0
                # if date_year != date_o.year:
                #     iyear += 1.0
                #     date_year = date_o.year
                #     # os.system(cmd.format(shp=file_shp, fi=file_i, fo=file_o))
                
                # Left, month - 1
                geo_trans, geo_proj, size_x, size_y = Open_array_info(file_i_l)

                ds = gdal.Open(file_i_l)
                ds_cols = int(np.min([ds_cols, ds.RasterXSize]))
                ds_rows = int(np.min([ds_rows, ds.RasterYSize]))
                
                ds_band = ds.GetRasterBand(1)
                ds_ndv = ds_band.GetNoDataValue()
                ds_data_l = ds_band.ReadAsArray()
                ds_data_l = np.where(ds_data_l == ds_ndv, 0.0, ds_data_l)
                
                ds = None

                # Right, month + 1
                geo_trans, geo_proj, size_x, size_y = Open_array_info(file_i_r)

                ds = gdal.Open(file_i_r)
                ds_cols = int(np.min([ds_cols, ds.RasterXSize]))
                ds_rows = int(np.min([ds_rows, ds.RasterYSize]))
                
                ds_band = ds.GetRasterBand(1)
                ds_ndv = ds_band.GetNoDataValue()
                ds_data_r = ds_band.ReadAsArray()
                ds_data_r = np.where(ds_data_r == ds_ndv, 0.0, ds_data_r)

                ds = None
                                
                if data is None:
                    data = np.zeros((ds_rows, ds_cols))
                data[0:ds_rows, 0:ds_cols] += (ds_data_r[0:ds_rows, 0:ds_cols] - ds_data_l[0:ds_rows, 0:ds_cols]) / 2.0

                file_o_ds = os.path.join(dir_in, 'dS-{}.tif'.format(fname_o[:-4]))
                Save_as_tiff(name=file_o_ds, data=data, geo=geo_trans, projection="WGS84")

                file_o_cutline = os.path.join(dir_out, '{}_cutline.tif'.format(fname_o[:-4]))
                os.system(cmd2.format(shp=file_shp, fi=file_o_ds, fo=file_o_cutline))
            else:
                if os.path.isfile(file_i_r) == False:
                    print(file_i_r)
                
        data = data / nmonth * 12.0
        Save_as_tiff(name=file_o, data=data, geo=geo_trans, projection="WGS84")

        file_o_mean_tmp = os.path.join(dir_out, '{}_yearly_tmp.tif'.format(fname_o.split('-')[0]))
        os.system(cmd1.format(fi=file_o, fo=file_o_mean_tmp))
   
        file_o_mean = os.path.join(dir_out, '{}_yearly.tif'.format(fname_o.split('_')[0]))
        os.system(cmd2.format(shp=file_shp, fi=file_o_mean_tmp, fo=file_o_mean))

        os.remove(file_o)
        os.remove(file_o_mean_tmp)


if __name__ == "__main__":
    # Warp(destNameOrDestDS, srcDSOrSrcDSTab)

    # WarpOptions()
    # https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
    # https://gdal.org/python/osgeo.gdal-pysrc.html#WarpOptions
    
    # gdal.Warp(r"%s" % filename, format='GTiff')
    
    path = os.path.join(
        os.getcwd(),
        os.path.dirname(
            inspect.getfile(
                inspect.currentframe())),
        '../'
    )
    
    dir_tmp = os.path.join(path, 'Data', 'Output', 'tmp')
    if not os.path.exists(dir_tmp):
        os.makedirs(dir_tmp)

    dir_out = os.path.join(path, 'Data', 'Output', 'tif')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)


    file_shp = os.path.join(path, 'Data', 'Shapefile', 'Mindanao-RiverBasin.shp')

    cmd1 = 'gdalwarp -of GTiff -overwrite -s_srs epsg:4326 -t_srs epsg:4326 -r near -tr 0.05 0.05 {fi} {fo}'
    cmd2 = 'gdalwarp -of GTiff -overwrite -s_srs epsg:4326 -t_srs epsg:4326 -r near -tr 0.05 0.05 -cutline {shp} -crop_to_cutline {fi} {fo}'

    date_s = '2014-01-01'
    date_e = '2019-12-31'

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
        main(dir_tmp, dir_out,
             prod_val['tif'], file_shp,
             cmd1, cmd2)
            