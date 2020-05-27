# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:08:29 2019

@author: sse
"""

import numpy as np
import datetime as dt
import os
import gdal
import netCDF4
import re

import time
start = time.time()


ds = gdal.Open(r"E:\sse\Test_nc\AET_WAPOR.v2.0_level2_mm-month-1_monthly_2009.01.tif")
a = ds.ReadAsArray()
band = ds.GetRasterBand(1)
ndv = band.GetNoDataValue()
print(ndv)

nlat, nlon = np.shape(a)

b = ds.GetGeoTransform()  # bbox, interval
lon = np.arange(nlon)*b[1]+b[0]
lat = np.arange(nlat)*b[5]+b[3]


basedate = dt.datetime(2009,1,1,0,0,0)

# create NetCDF file
nco = netCDF4.Dataset('AETI_2009_Nile.nc','w',clobber=True)

# chunking is optional, but can improve access a lot: 
# (see: http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes)
chunk_x=5000
chunk_y=5000
chunk_time=1

# create dimensions, variables and attributes:
nco.createDimension('lon', nlon)
nco.createDimension('lat', nlat)
nco.createDimension('time', None)
    
timeo = nco.createVariable('time','f4',('time'))
timeo.units = 'days since 2009-01-01 00:00'
timeo.standard_name = 'time'

lono = nco.createVariable('lon', 'f4', ('lon',))
lono.units = 'degree'
lono.standard_name = 'longitude'

lato = nco.createVariable('lat', 'f4', ('lat',))
lato.units = 'degree'
lato.standard_name = 'latitude'


# Create container variable for CRS: lon/lat WGS84 datum
crso = nco.createVariable('crs', 'i4')
crso.long_name = 'Lon/Lat Coords in WGS84'
crso.grid_mapping_name = 'latitude_longitude'
crso.longitude_of_prime_meridian = 0.0
crso.semi_major_axis = 6378137.0
crso.inverse_flattening = 298.257223563
    
# create  float variable for variable, with chunking
AETIo = nco.createVariable('AETI', 'i2',  ('time', 'lat', 'lon'), 
   zlib=True,complevel=9,chunksizes=[chunk_time,chunk_y,chunk_x],fill_value=-9999)
AETIo.units = 'mm/month'
AETIo.scale_factor = 0.1
AETIo.add_offset = 0
AETIo.grid_mapping = 'crs'
AETIo.long_name = 'WAPOR AETI'
AETIo.standard_name = 'AETI'
AETIo.set_auto_maskandscale(False)

nco.Conventions='CF-1.6'

# Write lon,lat
lono[:] = lon
lato[:] = lat

N = 1 ## number of files load at atime
pat = re.compile('AET_WAPOR.v2.0_level2_mm-month-1_monthly_[0-9]{4}\.[0-9]{2}')
itime=0
written = 0
data = np.zeros((N, nlat, nlon), dtype=np.int16)

#step through data, writing time and data to NetCDF
for root, dirs, files in os.walk('E:/sse/Test_nc/'):
    dirs.sort()
    files.sort()
    for f in files:
        if re.match(pat,f):
           
            # read the time values by parsing the filename
            year=int(f[41:45])
            mon=int(f[46:48])
            date=dt.datetime(year,mon,1,0,0,0)
            print(date)
            dtime=(date-basedate).total_seconds()/86400.
            timeo[itime]=dtime
           # min temp
            tmn_path = os.path.join(root,f)
            print(tmn_path)
            tmn=gdal.Open(tmn_path)
            a=tmn.ReadAsArray()  #data
            a = np.ma.masked_equal(a, ndv)
            #a[a == -9999.0] = np.nan
            a=np.round(a*10,0)
            data[itime%N, :, :] = a.astype(int) 
            
            #AETIo[itime,:,:]=a
            itime=itime+1
            if itime  % N == 0 and itime != 0:
                    AETIo[written:itime, :, :] = data
                    written = itime

nco.close()
print('It took', time.time()-start, 'seconds.')