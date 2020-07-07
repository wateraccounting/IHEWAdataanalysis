# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:08:29 2019

@author: sse
"""

import inspect
import os
from datetime import datetime

import numpy as np
import pandas as pd

import gdal
import netCDF4


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


def main(dir_in, dir_out, tif, nc):
    date_fmt = '%Y-%m-%d'
    
    for var in tif.keys():
        date_s = datetime.strptime(tif[var]['period']['s'], date_fmt)
        date_e = datetime.strptime(tif[var]['period']['e'], date_fmt)
        
        data = None
        ds_cols, ds_rows = np.inf, np.inf

        dates = pd.date_range(date_s, date_e, freq='MS')
        ntime = len(dates)

        fname_o = nc['name']
        file_o = os.path.join(dir_out, fname_o)
        
        # get GeoTiff meta data
        date = date_s
        date_year = date.year
        date_month = date.month
        date_day = date.day
    
        fname_i = tif[var]['output'].format(dtime=date)
        file_i = os.path.join(dir_in, fname_i)

        if os.path.isfile(file_i):
            geo_trans, geo_proj, size_x, size_y = Open_array_info(file_i)

            ds = gdal.Open(file_i)
            ds_cols = int(np.min([ds_cols, ds.RasterXSize]))
            ds_rows = int(np.min([ds_rows, ds.RasterYSize]))
            
            ds_band = ds.GetRasterBand(1)
            ds_ndv = ds_band.GetNoDataValue()
            ds_data = ds_band.ReadAsArray()
            ds_data = np.where(ds_data == ds_ndv, 0.0, ds_data)

            ds = None

            nlon = ds_cols
            nlat = ds_rows
            lon = np.arange(nlon)*geo_trans[1]+geo_trans[0]
            lat = np.arange(nlat)*geo_trans[5]+geo_trans[3]

            data = np.zeros((1, ds_rows, ds_cols))
            data[0, 0:ds_rows, 0:ds_cols] = ds_data[0:ds_rows, 0:ds_cols]
        else:
            print(file_i)

        # create NetCDF file
        nco = netCDF4.Dataset(file_o,'w',clobber=True)
        if data is not None:
            # create dimensions, variables and attributes:
            nco.createDimension('lon', nlon)
            nco.createDimension('lat', nlat)
            nco.createDimension('time', ntime)
                
            nco_time = nco.createVariable('time','f4',('time'))
            nco_time.units = 'days since {dtime:%Y}-01-01 00:00'.format(dtime=date_s)
            nco_time.standard_name = 'time'

            mco_lon = nco.createVariable('lon', 'f4', ('lon',))
            mco_lon.units = 'degree'
            mco_lon.standard_name = 'longitude'

            nco_lat = nco.createVariable('lat', 'f4', ('lat',))
            nco_lat.units = 'degree'
            nco_lat.standard_name = 'latitude'


            # Create container variable for CRS: lon/lat WGS84 datum
            nco_crs = nco.createVariable('crs', 'i4')
            nco_crs.long_name = 'Lon/Lat Coords in WGS84'
            nco_crs.grid_mapping_name = 'latitude_longitude'
            nco_crs.longitude_of_prime_meridian = 0.0
            nco_crs.semi_major_axis = 6378137.0
            nco_crs.inverse_flattening = 298.257223563
                
            # create  float variable for variable, with chunking
            nco_var = nco.createVariable(tif[var]['variable'], 'i2',  ('time', 'lat', 'lon'), zlib=True, complevel=9, fill_value=-9999)
            nco_var.units = 'mm/month'
            nco_var.scale_factor = 1.0
            nco_var.add_offset = 0
            nco_var.grid_mapping = 'crs'
            nco_var.long_name = tif[var]['variable']
            nco_var.standard_name = tif[var]['variable']
            nco_var.set_auto_maskandscale(False)

            nco.Conventions='CF-1.6'

            # Write lon,lat
            mco_lon[:] = lon
            nco_lat[:] = lat

        itime = 0
        for date in dates:
            date_year = date.year
            date_month = date.month
            date_day = date.day
        
            fname_i = tif[var]['output'].format(dtime=date)
            file_i = os.path.join(dir_in, fname_i)

            if os.path.isfile(file_i):
                geo_trans, geo_proj, size_x, size_y = Open_array_info(file_i)

                ds = gdal.Open(file_i)
                ds_cols = int(np.min([ds_cols, ds.RasterXSize]))
                ds_rows = int(np.min([ds_rows, ds.RasterYSize]))
                
                nlon = ds_cols
                nlat = ds_rows
                
                ds_band = ds.GetRasterBand(1)
                ds_ndv = ds_band.GetNoDataValue()
                ds_data = ds_band.ReadAsArray()
                ds_data = np.where(ds_data == ds_ndv, 0.0, ds_data)

                ds = None

                data = np.zeros((1, ds_rows, ds_cols))
                data[0, 0:ds_rows, 0:ds_cols] = ds_data[0:ds_rows, 0:ds_cols]
            else:
                print(file_i)
            
            if data is not None:
                #step through data, writing time and data to NetCDF
                dtime=(date-date_s).total_seconds()/86400.
                nco_time[itime]=dtime
                nco_var[itime:itime+1, :, :] = data
            
            itime += 1

        nco.close()


if __name__ == "__main__":
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

    dir_out = os.path.join(path, 'Data', 'Output', 'nc')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # cmd1 = 'gdal_translate -of netCDF -co "FOMRAT=NC4" {fi} {fo}'

    date_s = '2014-01-01'
    date_e = '2019-12-31'

    products = {
        'ETA-3':{
            'tif': {
                'ETA': {
                    'name': 'GLDAS_v2.1_mm.m_MS-{dtime:%Y%m}.tif',
                    'version': 'v2.1',
                    'resolution': 'monthly',
                    'variable': 'ETA',
                    'period': {
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'GLDAS_v2.1_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'ETA-GLDAS.csv'
            },
            'nc': {
                'name': 'ETA-GLDAS.nc'
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
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'MOD16A2_v6_mm.m_MS-{dtime:%Y%m}.tif'
                }
            },
            'csv': {
                'name': 'ETA-MOD16A2.csv'
            },
            'nc': {
                'name': 'ETA-MOD16A2.nc'
            }
        },
        'ETA-6':{
            'tif': {
                'ETA': {
                    'name': 'SSEBop_v4_mm.m_MS-{dtime:%Y%m}.tif',
                    'version': 'v4',
                    'resolution': 'monthly',
                    'variable': 'ETA',
                    'period': {
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'SSEBop_v4_mm.m_MS-{dtime:%Y%m}.tif'
                }
            },
            'csv': {
                'name': 'ETA-SSEBop.csv'
            },
            'nc': {
                'name': 'ETA-SSEBop.nc'
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
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'CHIRPS_v2.0_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'PCP-CHIRPS.csv'
            },
            'nc': {
                'name': 'PCP-CHIRPS.nc'
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
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'GPM_v6_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'PCP-GPM.csv'
            },
            'nc': {
                'name': 'PCP-GPM.nc'
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
                        's': date_s,
                        'e': date_e
                    },
                    'output': 'TRMM_v7_mm.m_MS-{dtime:%Y%m}.tif',
                }
            },
            'csv': {
                'name': 'PCP-TRMM.csv'
            },
            'nc': {
                'name': 'PCP-TRMM.nc'
            }
        },
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
            },
            'nc': {
                'name': 'EWH-CSR-v3.1.nc'
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
            },
            'nc': {
                'name': 'EWH-CSR-v3.2.nc'
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
            },
            'nc': {
                'name': 'EWH-GFZ-v3.1.nc'
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
            },
            'nc': {
                'name': 'EWH-GFZ-v3.2.nc'
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
            },
            'nc': {
                'name': 'EWH-JPL-v3.1.nc'
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
            },
            'nc': {
                'name': 'EWH-JPL-v3.2.nc'
            }
        },
    }
    
    for prod_key, prod_val in products.items():
        print(prod_key, prod_val['tif'])
        main(dir_tmp, dir_out, prod_val['tif'], prod_val['nc'])
            
