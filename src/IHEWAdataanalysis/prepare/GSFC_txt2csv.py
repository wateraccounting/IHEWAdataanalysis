# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:51:44 2018

Need to download global GRACE GFSC product from
https://ssed.gsfc.nasa.gov/grace/products.html
GSFC.glb.200301_201607_v02.3b-ICE6G - ASCII

url = 'https://earth.gsfc.nasa.gov/sites/default/files/neptune/grace/mascons_2.4/GSFC.glb.200301_201607_v02.4-ICE6G.h5'
remote_file = os.path.join(path, 'GSFC.glb.200301_201607_v02.4-ICE6G.h5')

@author: cmi001 

gdalsrsinfo.exe "EPSG:4326"

cd "D:\IHEProjects\20200218-Philippines\Code"
python GSFC_txt2csv.py
"""
import os
root = os.path.dirname(os.path.abspath(__file__))
os.chdir(root)


import csv
from osgeo import ogr
import shapefile
import pandas as pd
import numpy as  np
import GSFC_functions as gf
from matplotlib import pyplot as plt


def plot_shp(shp):
    sf = shapefile.Reader(shp)
    plt.figure()
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y)
    plt.show()

case_name = 'Mindanao-RiverBasin'  # 97530 km2

pathIn  = os.path.join(root, '..', 'Data')
pathOut = os.path.join(root, '..', 'Data', 'Output')

pathFig = os.path.join(pathOut, 'fig')
if not os.path.exists(pathFig):
    os.makedirs(pathFig)

pathCsv = os.path.join(pathOut, 'csv')
if not os.path.exists(pathCsv):
    os.makedirs(pathCsv)

pathShp = os.path.join(pathOut, 'shp')
if not os.path.exists(pathShp):
    os.makedirs(pathShp)

BUFFER_DIST = .71
# BUFFER_DIST = 0.0
# FIELD_NAME = 'ADM1_EN'
FIELD_NAME = 'Name'

BASIN_SHP  = os.path.join(pathIn, 'Shapefile', "{cn}.shp".format(cn=case_name))
OUT_CSV = os.path.join(pathCsv , '{}-GSFC-{}.csv')

BUFFER_SHP = os.path.join(pathShp, "{cn}_buffer.shp".format(cn=case_name))
MASCON_SHP = os.path.join(pathShp, "{cn}_GSFC.shp".format(cn=case_name))
print('BASIN_SHP    : "{}"'.format(BASIN_SHP))
print('OUT_CSV      : "{}"'.format(OUT_CSV))
print('BUFFER_SHP   : "{}"'.format(BUFFER_SHP))
print('MASCON_SHP   : "{}"'.format(MASCON_SHP))

MASCON_DATA_FOLDER = os.path.join(pathIn, 'GRACE', 'GSFC', "GSFC.glb.200301_201607_v02.4-ICE6G")
MASCON_INFOR = os.path.join(MASCON_DATA_FOLDER, 'mascon.txt')
MASCON_SOLUT = os.path.join(MASCON_DATA_FOLDER, 'solution.txt')
MASCON_DATES = os.path.join(MASCON_DATA_FOLDER, 'time.txt')
print('MASCON_INFOR : "{}"'.format(MASCON_INFOR))
print('MASCON_SOLUT : "{}"'.format(MASCON_SOLUT))
print('MASCON_DATES : "{}"'.format(MASCON_DATES))

print('GRACE, create_buffer')
gf.create_buffer(BASIN_SHP, BUFFER_SHP, BUFFER_DIST, FIELD_NAME)

print('load, MASCON_INFOR, mascon.txt')
df_info = pd.read_csv(MASCON_INFOR, sep=r"\s+", header=None, skiprows=14,engine='python')
mascon_coords = zip(df_info[1], df_info[0])
mascon_area_km = df_info[5]

print('load, MASCON_DATES')
df_dates = pd.read_csv(MASCON_DATES, sep=r"\s+", header=None, skiprows=13,engine='python')
fract_dates = df_dates[2]
mascon_dates = [str(gf.convert_partial_year(fdate)) for fdate in fract_dates]
mascon_dates_daily = pd.date_range(start=mascon_dates[0],end=mascon_dates[-1],freq='D')

print('points_in_polygon, BUFFER_SHP')
# Return null geometry sometimes? Shell is not a LinearRing
# TODO-END, 20200221, QPan, wkbMultiPolygon.wkbPolygon.wkbLinearRing 
index_mascons_of_interest = gf.points_in_polygon(BUFFER_SHP, mascon_coords, pathFig)
# print('index_mascons_of_interest', index_mascons_of_interest)

print('load, MASCON_SOLUT, solution.txt')
data_lines = []
area_lines = []
with open(MASCON_SOLUT) as fp:
    for i, line in enumerate(fp):
        # >>> print(i, line[0:10])
        # 41173 13.08760 2
        # 41174 12.42396 2
        if i in np.array(index_mascons_of_interest) + 7:
            data_lines.append(np.array(line.rstrip('\n').rstrip().split(' ')).astype(float))
            area_lines.append(mascon_area_km[i])

print('create, MASCON_SHP of mascon areas')
# Adapeted from bec's SortGRACE.py
w = shapefile.Writer(MASCON_SHP, shapeType=shapefile.POLYGON)
w.field('MASCON_ID', 'C', '40')

for mascon_index in index_mascons_of_interest:
    ID = mascon_index+1
    lon_center = df_info[1][mascon_index]
    lat_center = df_info[0][mascon_index]
    lon_span = df_info[3][mascon_index]
    lat_span = df_info[2][mascon_index]
    w.poly([
            [[lon_center + .5 * lon_span, lat_center + .5 * lat_span],
             [lon_center - .5 * lon_span, lat_center + .5 * lat_span],
             [lon_center - .5 * lon_span, lat_center - .5 * lat_span],
             [lon_center + .5 * lon_span, lat_center - .5 * lat_span],
             [lon_center + .5 * lon_span, lat_center + .5 * lat_span]]
            ])
    w.record(ID,'Polygon')
w.close()

print('Get weights from relative intersection area')
basin_poly  = ogr.Open(BASIN_SHP)
mascon_poly = ogr.Open(MASCON_SHP)

basin_lyr  = basin_poly.GetLayer()
mascon_lyr = mascon_poly.GetLayer()

print('loop BASIN_SHP basin_lyr, MASCON_SHP mascon_lyr')
i_b_feature = 0
int_area_total = np.zeros(basin_lyr.GetFeatureCount())
basin_lyr_def = basin_lyr.GetLayerDefn()
basin_lyr_def_n = basin_lyr_def.GetFieldCount()
for b_feature in basin_lyr:
    b_geom = b_feature.GetGeometryRef()
    # FIELD_NAME = 'ADM1_EN'
    print('basin', (i_b_feature + 1), '/', basin_lyr.GetFeatureCount(), b_geom.GetGeometryName(), b_geom.Centroid().ExportToWkt())
    for i in range(basin_lyr_def_n):
        if basin_lyr_def.GetFieldDefn(i).GetName() == FIELD_NAME:
            print('     ',b_feature.GetField(FIELD_NAME))
    # continue
    
    ids = []
    int_area = []
    int_area_total[i_b_feature] = 0.0

    i_m_feature = 0
    for m_feature in mascon_lyr:
        m_geom = m_feature.GetGeometryRef()
        int_gemo = b_geom.Intersection(m_geom)
        
        ids.append(m_feature.GetField(0))
        int_area.append(int_gemo.GetArea())
        int_area_total[i_b_feature] += int_gemo.GetArea()
        # print('\t + int_area_total[{}.{}]: '.format(i_b_feature, i_m_feature), int_area_total[i_b_feature])
        
        if (i_m_feature + 1) % 100 == 0:
            print('\t', 'mascon', (i_m_feature + 1), '/', mascon_lyr.GetFeatureCount(), m_geom.GetGeometryName(), m_geom.Centroid().ExportToWkt())
            print('\t', '\ttotal_area: ', int_area_total[i_b_feature])
        i_m_feature += 1
    # reset the read position to the start
    mascon_lyr.ResetReading()
    print('\tint_area_total[{}]: '.format(i_b_feature), int_area_total[i_b_feature])
        
    print('\tcalculate weights')
    # TODO-END, 20200221, QPan, is it correct? Yes
    weights = np.array(int_area) / np.sum(int_area_total[i_b_feature])
    # print('\tweights', weights)
    # print('\tsum(weights)', np.sum(weights))

    print('\tcalculate weighted mascon')
    weighted_line = [data_lines[i] * weights[i] for i in range(len(data_lines))]
    # print('\tweighted_line: ', weighted_line)
    EWH = np.sum(weighted_line, 0)
    EWH = EWH * 10.0  # cm to mm
    print('\tEWH max: ', np.max(EWH), 'min: ', np.min(EWH))

    df_EWH = pd.DataFrame(index=pd.to_datetime(mascon_dates), data=np.array(EWH))
    df_EWH.to_csv(OUT_CSV.format('EWH', '{}'.format(i_b_feature)),
                  index_label='date', header=['Equivalent Water Height [mm]'],
                  date_format='%Y-%m-%d',  decimal='.', encoding='utf-8')
    
    print('\tcalculate storage change')
    # GRACE, 35 * 12453 km2 = 435855 km2
    # Wiki, 97530 km2
    # Shapefile, sum([‭12394.541633912566,
    #                 12399.979617052235,
    #                 12402.623636065848,
    #                 12398.418481014005,
    #                 12397.502652393287,
    #                 12401.18128282408]) = ‭74394.24730326202
    # weighted_area = [area_lines[i] * weights[i] for i in range(len(area_lines))]
    # weighted_total_Area = np.sum(weighted_area, 0)
    # print('\tweighted_total_Area km2:', weighted_total_Area)
    df_dS = pd.DataFrame(index=mascon_dates_daily)
    # Assign EWH to dS based on Timestamp
    df_dS = pd.merge(df_dS, df_EWH, left_index=True,right_index=True, how='outer')
    # Interpolate GRACE estimates to daily time-step
    df_dS = df_dS.interpolate()

    # 1. Calculate central difference
    df_dS_ctld = df_dS.resample('MS').first()
    df_dS_ctld = df_dS_ctld.diff(2).shift(-1)/2.
    # df_dS_ctld.to_csv(OUT_CSV.format('dS', '{}_{}'.format(i_b_feature, 'CtlDif')),
    df_dS_ctld.to_csv(OUT_CSV.format('dS', '{}'.format(i_b_feature)),
                      index_label='date', header=[str(i_b_feature)],
                      date_format='%Y-%m',  decimal='.', encoding='utf-8')
        
    # 2. Calculate monthly difference
    df_dS_dif1 = df_dS.resample('MS').first()
    df_dS_dif1 = df_dS_dif1.diff(1).shift(-1)
    df_dS_dif1.to_csv(OUT_CSV.format('dS', '{}_{}'.format(i_b_feature, 'Dif1M')),
                      index_label='date', header=[str(i_b_feature)],
                      date_format='%Y-%m',  decimal='.', encoding='utf-8')

    # 3. Calculate monthly difference
    df_dS_mean = df_dS.resample('MS').mean()
    df_dS_mean.to_csv(OUT_CSV.format('dS', '{}_{}'.format(i_b_feature, 'Mean')),
                      index_label='date', header=[str(i_b_feature)],
                      date_format='%Y-%m',  decimal='.', encoding='utf-8')
    
    i_b_feature += 1

