# -*- coding: utf-8 -*-

import pytest

import inspect
import os

import IHEWAdataanalysis


if __name__ == "__main__":
    print('\nAnalysis\n=====')
    path = os.path.join(
        os.getcwd(),
        os.path.dirname(
            inspect.getfile(
                inspect.currentframe()))
    )
    os.chdir(path)

    analysis = IHEWAdataanalysis.Analysis(path, 'test_analysis.yml')
    print(analysis._Analysis__conf)
