# -*- coding: utf-8 -*-
"""

`example
<https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py>`_

"""
import inspect
import os
import yaml

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import netCDF4

try:
    import ogr, osr, gdal
except ImportError:
    from osgeo import ogr, osr, gdal

try:
    # IHEClassInitError, IHEStringError, IHETypeError, IHEKeyError, IHEFileError
    from .exception import IHEClassInitError
except ImportError:
    from IHEWAdataanalysis.exception import IHEClassInitError


class Template(object):
    """This Base class

    Load base.yml file.

    Args:
        conf (dict): User defined configuration data from yaml file.
    """
    def __init__(self, conf):
        """Class instantiation
        """
        template = 'FAO.yml'
        path = os.path.join(
            os.getcwd(),
            os.path.dirname(
                inspect.getfile(
                    inspect.currentframe()))
        )
        self.path = ''
        self.data = {}
        self.doc = None
        self.__conf = conf

        data = self._conf(path, template)
        if len(data.keys()) > 0:
            self.path = path
            self.data = data
        else:
            raise IHEClassInitError(template) from None

        obj = self.create()
        if obj is not None:
            self.fig = obj

            print('\nFigure Start')
            print('Create temp dir:')

            print('>>>>>')
            # doc Cover
            self.plot('CoverPage')

    def _conf(self, path, template) -> dict:
        data = {}

        file_conf = os.path.join(path, template)
        with open(file_conf) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)

        return data

    def create(self) -> object:
        figure = self.data['figure']
        obj = None

        return obj

    def plot(self):
        pass

    def plot_line(self):
        pass

    def saveas(self):
        pass

    def close(self):
        pass