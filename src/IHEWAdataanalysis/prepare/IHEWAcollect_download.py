# -*- coding: utf-8 -*-
"""
pip install IHEWAcollect
cd "D:\IHEProjects\20200218-Philippines\Code"
python IHEWAcollect_download.py
pip uninstall IHEWAcollect
"""
import inspect
import os

import IHEWAcollect
import yaml


def main(path, test_args):
    from pprint import pprint

    # Download __init__
    for key, value in test_args.items():
        print('\n{:>4s}'
              '{:>20s}{:>6s}{:>20s}{:>20s}{:>20s}\n'
              '{:->90s}'.format(key,
                                value['product'],
                                value['version'],
                                value['parameter'],
                                value['resolution'],
                                value['variable'],
                                '-'))

        IHEWAcollect.Download(workspace=path,
                              product=value['product'],
                              version=value['version'],
                              parameter=value['parameter'],
                              resolution=value['resolution'],
                              variable=value['variable'],
                              bbox=value['bbox'],
                              period=value['period'],
                              nodata=value['nodata'],
                              is_status=True,
                              is_save_temp=False,
                              is_save_remote=False)


if __name__ == "__main__":
    path = os.path.join(
        os.getcwd(),
        os.path.dirname(
            inspect.getfile(
                inspect.currentframe())),
        '../', 'Data'
    )
    
    period = {
        's': '2014-01-01',
        'e': '2019-12-31'
    }
    bbox = {
        'w': 118.0642363480000085,
        'n':  10.4715946960000679,
        'e': 126.6049655970000458,
        's':   4.5872944970000731
    }

    test_args = {
        # # evapotranspiration, ETA, daily
        # 'ALEXI-daily': {
        #     'product': 'ALEXI',
        #     'version': 'v1',
        #     'parameter': 'evapotranspiration',
        #     'resolution': 'daily',
        #     'variable': 'ETA',
        #     'bbox': bbox,
        #     'period': period,
        #     'nodata': -9999
        # },
        
        # # evapotranspiration, ETA, monthly
        # 'CMRSET-monthly': {
        #     'product': 'CMRSET',
        #     'version': 'v1',
        #     'parameter': 'evapotranspiration',
        #     'resolution': 'monthly',
        #     'variable': 'ETA',
        #     'bbox': bbox,
        #     'period': period,
        #     'nodata': -9999
        # },
        'GLDAS-monthly': {
            'product': 'GLDAS',
            'version': 'v2.1',
            'parameter': 'evapotranspiration',
            'resolution': 'monthly',
            'variable': 'ETA',
            'bbox': bbox,
            'period': period,
            'nodata': -9999
        },
        # 'GLEAM-monthly': {
        #     'product': 'GLEAM',
        #     'version': 'v3.3b',
        #     'parameter': 'evapotranspiration',
        #     'resolution': 'monthly',
        #     'variable': 'ETA',
        #     'bbox': bbox,
        #     'period': period,
        #     'nodata': -9999
        # },
        'MOD16A2-eight_daily': {
            'product': 'MOD16A2',
            'version': 'v6',
            'parameter': 'evapotranspiration',
            'resolution': 'eight_daily',
            'variable': 'ETA',
            'bbox': bbox,
            'period': period,
            'nodata': -9999
        },
        'SSEBop-monthly': {
            'product': 'SSEBop',
            'version': 'v4',
            'parameter': 'evapotranspiration',
            'resolution': 'monthly',
            'variable': 'ETA',
            'bbox': bbox,
            'period': period,
            'nodata': -9999
        },
        
        # precipitation, PCP, daily
        
        # precipitation, PCP, monthly
        'CHIRPS-monthly': {
            'product': 'CHIRPS',
            'version': 'v2.0',
            'parameter': 'precipitation',
            'resolution': 'monthly',
            'variable': 'PCP',
            'bbox': bbox,
            'period': period,
            'nodata': -9999
        },
        'GPM-monthly': {
            'product': 'GPM',
            'version': 'v6',
            'parameter': 'precipitation',
            'resolution': 'monthly',
            'variable': 'PCP',
            'bbox': bbox,
            'period': period,
            'nodata': -9999
        },
        'TRMM-v7a-monthly': {
            'product': 'TRMM',
            'version': 'v7a',
            'parameter': 'precipitation',
            'resolution': 'monthly',
            'variable': 'PCP',
            'bbox': bbox,
            'period': {
                's': period['s'],
                'e': '2010-09-30'
            },
            'nodata': -9999
        },
        'TRMM-v7-monthly': {
            'product': 'TRMM',
            'version': 'v7',
            'parameter': 'precipitation',
            'resolution': 'monthly',
            'variable': 'PCP',
            'bbox': bbox,
            'period': {
                's': '2010-10-01',
                'e': period['e']
            },
            'nodata': -9999
        },
        
        # NDVI
        # 'MOD13Q1-sixteen_daily': {
        #     'product': 'MOD13Q1',
        #     'version': 'v6',
        #     'parameter': 'land',
        #     'resolution': 'sixteen_daily',
        #     'variable': 'NDVI',
        #     'bbox': bbox,
        #     'period': period,
        #     'nodata': -9999
        # },
        # 'PROBAV-daily': {
        #     'product': 'PROBAV',
        #     'version': 'v1.01',
        #     'parameter': 'land',
        #     'resolution': 'daily',
        #     'variable': 'NDVI',
        #     'bbox': bbox,
        #     'period': period,
        #     'nodata': -9999
        # },

        # GRACE
        'CSR-daily-v3.1': {
            'product': 'CSR',
            'version': 'v3.1',
            'parameter': 'grace',
            'resolution': 'daily',
            'variable': 'EWH',
            'bbox': bbox,
            'period': {
                's': '2013-11-01',
                'e': '2018-01-01'
            },
            'nodata': -9999
        },
        'CSR-daily-v3.2': {
            'product': 'CSR',
            'version': 'v3.2',
            'parameter': 'grace',
            'resolution': 'daily',
            'variable': 'EWH',
            'bbox': bbox,
            'period': {
                's': '2017-12-01',
                'e': '2020-01-01'
            },
            'nodata': -9999
        },
        'GFZ-daily-v3.1': {
            'product': 'GFZ',
            'version': 'v3.1',
            'parameter': 'grace',
            'resolution': 'daily',
            'variable': 'EWH',
            'bbox': bbox,
            'period': {
                's': '2013-11-01',
                'e': '2018-01-01'
            },
            'nodata': -9999
        },
        'GFZ-daily-v3.2': {
            'product': 'GFZ',
            'version': 'v3.2',
            'parameter': 'grace',
            'resolution': 'daily',
            'variable': 'EWH',
            'bbox': bbox,
            'period': {
                's': '2017-12-01',
                'e': '2020-01-01'
            },
            'nodata': -9999
        },
        'JPL-daily-v3.1': {
            'product': 'JPL',
            'version': 'v3.1',
            'parameter': 'grace',
            'resolution': 'daily',
            'variable': 'EWH',
            'bbox': bbox,
            'period': {
                's': '2013-11-01',
                'e': '2018-01-01'
            },
            'nodata': -9999
        },
        'JPL-daily-v3.2': {
            'product': 'JPL',
            'version': 'v3.2',
            'parameter': 'grace',
            'resolution': 'daily',
            'variable': 'EWH',
            'bbox': bbox,
            'period': {
                's': '2017-12-01',
                'e': '2020-01-01'
            },
            'nodata': -9999
        },
    }

    main(path, test_args)
