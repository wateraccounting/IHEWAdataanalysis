# -*- coding: utf-8 -*-
import inspect
import os

import IHEWAreport


if __name__ == "__main__":
    print('\nReport\n=====')
    path = os.path.join(
        os.getcwd(),
        os.path.dirname(
            inspect.getfile(
                inspect.currentframe()))
    )
    os.chdir(path)

    report = IHEWAreport.Report(path, 'test_report.yml')

    # report.get_config()
    # print(report._Report__conf['path'], '\n',
    #       report._Report__conf['name'], '\n',
    #       report._Report__conf['time'], '\n',
    #       report._Report__conf['data'])
