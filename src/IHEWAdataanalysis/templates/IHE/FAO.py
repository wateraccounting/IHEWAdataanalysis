# -*- coding: utf-8 -*-
"""

`example
<https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py>`_

"""
import inspect
import os
import yaml

import numpy as np
import pandas as pd

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
        print('{}.{}'.format(path, template))

    def _conf(self, path, template):
        pass

    def create(self):
        pass

    def write(self):
        pass

    def saveas(self):
        pass

    def close(self):
        pass