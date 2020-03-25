# -*- coding: utf-8 -*-
"""
IHEWAdataanalysis: IHE Water Accounting Data Analysis Tools
"""


from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'IHEWAdataanalysis'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

try:
    from .report import Report
except ImportError:
    from IHEWAdataanalysis.analysis import Analysis
__all__ = ['Analysis']

# TODO, 20190931, QPan,
