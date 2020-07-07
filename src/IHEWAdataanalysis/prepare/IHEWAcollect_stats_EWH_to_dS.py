# -*- coding: utf-8 -*-
"""
cd "D:\IHEProjects\20200218-Philippines\Code"
python IHEWAcollect_stats_EWH_to_dS.py
"""
import inspect
import os

import csv
import pandas as pd
import numpy as  np
from matplotlib import pyplot as plt

# nfeature = 8
ifeature = 1
date_s = '2013-12-01'
date_e = '2020-01-01'

mascon_dates_daily = pd.date_range(start=date_s, end=date_e, freq='D')

# path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(
    os.getcwd(),
    os.path.dirname(
        inspect.getfile(
            inspect.currentframe())),
    '../'
)
dir_csv = os.path.join(path, 'Data', 'Output', 'csv')

In_NAMEs = ['CSR', 'GFZ', 'JPL']
In_VERSIONs = ['v3.1', 'v3.2']

for In_VERSION in In_VERSIONs:
    for In_NAME in In_NAMEs:
        print(In_NAME)
        In_CSV = os.path.join(dir_csv, 'EWH-{}-{}.csv'.format(In_NAME, In_VERSION))  # unit: cm

        # for ifeature in range(nfeature):
        OUT_CSV = os.path.join(dir_csv, 'dS-{}-{}-{}.csv'.format(In_NAME, In_VERSION, ifeature))
        col_name = 'mean_{}'.format(ifeature)

        df_EWH = pd.read_csv(In_CSV, index_col='date', parse_dates=True, na_values='None')
        print(df_EWH.describe())


        # df_dS = pd.DataFrame(index=mascon_dates_daily, columns=[str(i) for i in range(nfeature)])
        # df_dS_ctld = pd.DataFrame(index=mascon_dates_daily, columns=[str(i) for i in range(nfeature)])
        df_dS = pd.DataFrame(index=mascon_dates_daily, columns=[str(ifeature)])
        df_dS_ctld = pd.DataFrame(index=mascon_dates_daily, columns=[str(ifeature)])

        tmp_df_dS = pd.DataFrame(index=mascon_dates_daily)

        EWH = df_EWH[col_name].values
        EWH = EWH * 1.0  # mm to mm
        print('\tcalculate storage change {}'.format(ifeature))

        tmp_df_EWH = pd.DataFrame(index=pd.to_datetime(df_EWH.index), data=np.array(EWH))

        df_dS = pd.DataFrame(index=mascon_dates_daily)
        df_dS = pd.merge(df_dS, tmp_df_EWH, left_index=True,right_index=True, how='outer')
        df_dS = df_dS.interpolate()

        # 1. Calculate central difference
        df_dS_ctld = df_dS.resample('MS').first()
        df_dS_ctld = df_dS_ctld.diff(2).shift(-1)/2.
        
        df_dS_ctld = df_dS_ctld.drop([mascon_dates_daily[0],mascon_dates_daily[-1]])

        df_dS_ctld.to_csv(OUT_CSV,
                          index_label='date', header=[col_name],
                          date_format='%Y-%m',  decimal='.', encoding='utf-8')
    
