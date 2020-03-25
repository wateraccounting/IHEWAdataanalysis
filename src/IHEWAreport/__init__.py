# -*- coding: utf-8 -*-
"""
IHEWAreport: IHE Water Accounting Report Tools
"""


from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = 'IHEWAreport'
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

try:
    from .report import Report
except ImportError:
    from IHEWAreport.report import Report
__all__ = ['Report']

# TODO, 20190931, QPan,
