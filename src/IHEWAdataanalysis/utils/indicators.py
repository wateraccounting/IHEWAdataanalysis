# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:11:53 2019

@author: bec
"""
import xarray as xr
from scipy import stats
import numpy as np


def bias(DS1, DS2, min_samples):
    x = DS1.where(DS2.notnull())
    y = DS2.where(DS1.notnull())

    n = x.notnull().sum(dim="time")

    sum1 = x.sum(dim="time", skipna=True)
    sum2 = y.sum(dim="time", skipna=True)

    b = sum2 / sum1

    return b.where(n >= min_samples)


def nash_sutcliffe(DS1, DS2, min_samples):
    x = DS1.where(DS2.notnull())
    y = DS2.where(DS1.notnull())

    n = x.notnull().sum(dim="time")

    part1 = ((y - x) ** 2).sum(dim="time", skipna=True)
    part2 = ((x - x.mean(dim="time", skipna=True)) ** 2).sum(dim="time", skipna=True)
    ns = 1. - part1 / part2

    return ns.where(n >= min_samples)


def MAE(DS1, DS2, min_samples):
    x = DS1.where(DS2.notnull())
    y = DS2.where(DS1.notnull())

    n = x.notnull().sum(dim="time")

    mae = xr.ufuncs.fabs(y - x).mean(dim="time", skipna=True)

    return mae.where(n >= min_samples)


def pearson_correlation(DS1, DS2, min_samples, dim="time", confidence=0.05, tails=2):
    x = DS1.where(DS2.notnull())
    y = DS2.where(DS1.notnull())

    n = x.notnull().sum(dim=dim)
    df = n - 2

    part1 = x - x.mean(dim=dim, skipna=True)
    part2 = y - y.mean(dim=dim, skipna=True)

    r = (part1 * part2).sum(dim=dim, skipna=True) / \
        (xr.ufuncs.sqrt((part1 ** 2).sum(dim=dim, skipna=True)) *
         xr.ufuncs.sqrt((part2 ** 2).sum(dim=dim, skipna=True)))
    t = (r * xr.ufuncs.sqrt(df)) / \
        xr.ufuncs.sqrt(1 - r ** 2)

    t_crit = xr.apply_ufunc(stats.t.ppf, 1 - (confidence / tails),
                            df, dask="allowed")

    pos_sig = xr.concat([r > 0, t > t_crit], dim="new").all(dim="new")
    neg_sig = xr.concat([r < 0, xr.ufuncs.fabs(t) > t_crit], dim="new").all(dim="new")

    significance = xr.concat([pos_sig, neg_sig], dim="new").any(dim="new")

    significance.attrs['confidence_level'] = confidence
    significance.attrs["tails"] = tails

    return r.where(n >= min_samples), significance.where(n >= min_samples)


# def RMSE(DS1, DS2, min_samples):
#     x = DS1.where(DS2.notnull())
#     y = DS2.where(DS1.notnull())
#
#     n = x.notnull().sum(dim="time")
#
#     rmse = xr.ufuncs.sqrt(((y - x) ** 2).mean(dim="time", skipna=True))
#
#     return rmse.where(n >= min_samples)
def RMSE(pred, targ):
    # x = DS1.where(DS2.notnull())
    # y = DS2.where(DS1.notnull())
    #
    # n = x.notnull().sum(dim="time")

    rmse = np.sqrt(np.mean((pred-targ)**2))

    return rmse


def NRMSE(DS1, DS2, min_samples):
    x = DS1.where(DS2.notnull())
    y = DS2.where(DS1.notnull())

    n = x.notnull().sum(dim="time")

    rmse = xr.ufuncs.sqrt(((y - x) ** 2).mean(dim="time", skipna=True)) / \
        (y.max(dim="time") - y.min(dim="time"))

    return rmse.where(n >= min_samples)




