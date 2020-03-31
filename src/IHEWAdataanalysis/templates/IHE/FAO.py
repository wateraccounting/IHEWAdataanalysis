# -*- coding: utf-8 -*-
"""

`example
<https://matplotlib.org/3.2.0/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py>`_

# df['c'] = df[['y', 'x']].apply(lambda row: abs(row['y'] / row['x'] - 1.0), axis=1)

"""
import inspect
import os
import yaml

import fnmatch
import itertools
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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

try:
    from .indicators import RMSE
except ImportError:
    from IHEWAdataanalysis.indicators import RMSE


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

        self.allow_ftypes = {
            'CSV': 'csv'
        }
        self.allow_ptypes = [
            'bar_wb',
            'heatmap',
            'line_prod',
            'scatter_prod'
        ]
        self.path = path
        self.workspace = ''
        self.conf = None
        self.data = None
        self.ifeature = 1

        self.__conf = conf

        conf = self._conf(path, template)
        if len(conf.keys()) > 0:
            self.conf = conf
        else:
            raise IHEClassInitError(template) from None
        print(self.conf.keys())

        data = self._data(self.__conf['path'])
        if len(data.keys()) > 0:
            self.data = data
        else:
            raise IHEClassInitError(template) from None
        print(self.data.keys())

        print('\nFigure Start')
        self.workspace = os.path.join(self.__conf['path'], 'IHEWAdataanalysis', 'fig')
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

        print('>>>>>')
        for fig_name, fig_obj in self.create().items():
            # print('{} {} "{}"'.format(fig_obj['obj'].number,
            #                           fig_name,
            #                           fig_obj['obj'].get_size_inches() *
            #                           fig_obj['obj'].dpi))
            ptype = fig_obj['ptype']
            if ptype in self.allow_ptypes:
                if ptype =='line_prod':
                    self.plot_line_prod(fig_name)
                if ptype == 'heatmap':
                    self.plot_heatmap(fig_name)
                if ptype == 'scatter_prod':
                    self.plot_scatter_prod(fig_name)
                if ptype == 'bar_wb':
                    self.plot_bar_wb(fig_name)
            else:
                print('Warning "{}" not support.'.format(
                    ptype
                ))

    def _conf(self, path, template) -> dict:
        conf = {}

        file_conf = os.path.join(path, template)
        with open(file_conf) as fp:
            conf = yaml.load(fp, Loader=yaml.FullLoader)

        return conf

    def _data(self, path) -> dict:
        data = {}

        try:
            conf_data = self.__conf['data']['data']
        except KeyError:
            data = {}
        else:
            for variable, products in conf_data.items():
                data[variable] = {}

                for prod_name, prod_conf in products.items():
                    try:
                        folder = os.path.split(prod_conf['folder'])
                        ftype = prod_conf['ftype']
                    except KeyError:
                        print('No "folder" in yaml.')
                        data = {}
                    else:
                        if ftype in self.allow_ftypes.keys():
                            fname_ptn = '{var}-{pod}.{typ}'.format(
                                var=variable,
                                pod=prod_name,
                                typ=self.allow_ftypes[ftype]
                            )
                            for root, dnames, fnames in os.walk(os.path.join(path,
                                                                             *folder)):
                                for fname in fnmatch.filter(fnames, fname_ptn):
                                    file = os.path.join(root, fname)
                                    print('Loading{:>10s}{:>20s} "{}"'.format(variable,
                                                                              prod_name,
                                                                              file))
                                    data[variable][prod_name] = pd.read_csv(file)
                                    # print('{}'.format(
                                    #     data[variable][prod_name].describe()))
                        else:
                            data = {}

        return data

    def create(self) -> dict:
        objs = {}

        for fig_name, fig_conf in self.conf.items():
            objs[fig_name] = fig_conf
            objs[fig_name]['obj'] = plt.figure(**fig_conf['figure'])

        return objs

    def plot_line_prod(self, name):
        fig_conf = self.conf[name]
        # parse yaml
        fig_data = self.data[fig_conf['data']]

        prod_names = list(fig_data.keys())
        prod_nprod = len(prod_names)

        if prod_nprod > 0:
            fig_title = fig_conf['title']
            print(fig_title)

            fig_nplt = prod_nprod
            fig_ncol = int(np.floor(np.sqrt(fig_nplt)))
            fig_nrow = int(np.ceil(float(fig_nplt) / float(fig_ncol)))

            fig = fig_conf['obj']
            fig.suptitle(fig_title)
            axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)

            ax_ticksize = 4
            ax_labelsize = 4

            # fig_comb = []
            # for i in range(prod_nprod):
            #     for j in range(prod_nprod):
            #         if j > i:
            #             fig_comb.append([i, j])

            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j
                    if iplt < fig_nplt:
                        prod_name = prod_names[iplt]
                        print('{:>10d}{:>20s}'.format(iplt, prod_name))

                        # prepare data
                        xlabel = 'date'
                        ylabel = '{}'.format(prod_name)

                        # x = pd.DataFrame(fig_data[xlabel],
                        #                  columns=['date', '{}'.format(self.ifeature)])
                        df = pd.DataFrame(fig_data[ylabel],
                                          columns=['date', '{}'.format(self.ifeature)])

                        # calculate data
                        # print(df)

                        # plot data
                        df['date'] = pd.to_datetime(df['date'], format='%Y-%m')

                        axes[i, j].plot(df['date'], df['{}'.format(self.ifeature)],
                                        color='black',
                                        linestyle='solid',
                                        linewidth=1)
                        axes[i, j].xaxis.set_major_locator(mdates.YearLocator())
                        axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                        axes[i, j].tick_params(axis='x', which='major', direction='in',
                                               pad=0.5, length=1,
                                               labelsize=ax_ticksize,
                                               labelrotation=45)
                        axes[i, j].tick_params(axis='y', which='major', direction='in',
                                               pad=0.5, length=1,
                                               labelsize=ax_ticksize,
                                               labelrotation=90)
                        axes[i, j].set_xlabel(xlabel,
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                        axes[i, j].set_ylabel(ylabel,
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                    else:
                        axes[i, j].remove()

            # fig.autofmt_xdate()
            fig.subplots_adjust(bottom=0.05, top=0.9,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.4)
            self.saveas(fig, name)
            self.close(fig)

    def plot_heatmap(self, name):
        fig_conf = self.conf[name]
        # parse yaml
        fig_data = {}
        var_names = []
        var_opers = []
        tmp_names_root = fig_conf['data'].split('.')
        iagn = 0
        for variable_root in tmp_names_root:
            if len(variable_root.split('-')) > 1:
                tmp_names = variable_root.split('-')
                isub = 0
                for variable in tmp_names:
                    if len(variable.split('+')) > 1:
                        ipls = 0
                        for sub_var in variable.split('+'):
                            if ipls > 0:
                                var_opers.append('+')
                            else:
                                var_opers.append('-')
                            var_names.append(sub_var)
                            fig_data[sub_var] = self.data[sub_var]
                            ipls += 1
                    else:
                        if isub > 0:
                            var_opers.append('-')
                        var_names.append(variable)
                        fig_data[variable] = self.data[variable]
                    isub += 1
            else:
                if iagn > 0:
                    var_opers.append('.')
                var_names.append(variable_root)
                fig_data[variable_root] = self.data[variable_root]
            iagn += 1
        # print(var_names, var_opers)

        fig_title = fig_conf['title']
        print(fig_title)

        fig_nplt = 1
        prod_names = []
        prod_nprod = []
        for ivar in range(len(var_names)):
            prod_names.append(list(fig_data[var_names[ivar]].keys()))
            prod_nprod.append([i for i in range(len(fig_data[var_names[ivar]].keys()))])
            fig_nplt *= np.prod(len(fig_data[var_names[ivar]].keys()))

        fig_nrow = int(np.floor(np.sqrt(fig_nplt)))
        fig_ncol = int(np.ceil(float(fig_nplt) / float(fig_nrow)))

        fig = fig_conf['obj']
        fig.suptitle(fig_title)
        axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)

        ax_titlesize = 6
        ax_ticksize = 4
        ax_labelsize = 4
        ax_legendsize = 3

        fig_comb = list(itertools.product(*prod_nprod))
        # print(len(fig_comb), prod_names)
        if len(fig_comb) > 0:
            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j
                    if iplt < fig_nplt:
                        prod_list = list(fig_comb[iplt])

                        # prepare data
                        ivar = 0
                        var_name = var_names[ivar]
                        prod_name = prod_names[ivar][prod_list[ivar]]

                        xlabel = 'date'
                        ylabel = '{}'.format(prod_name)

                        df_col = ['date']
                        df = pd.DataFrame(fig_data[var_name][prod_name],
                                          columns=['date',
                                                   '{}'.format(self.ifeature)])
                        df_col.append('var_{}'.format(ivar))
                        for ioper in range(len(var_opers)):
                            ivar = ioper + 1
                            var_oper = var_opers[ioper]
                            var_name = var_names[ivar]
                            prod_name = prod_names[ivar][prod_list[ivar]]

                            ylabel += '{}{}'.format(var_oper, prod_name)
                            df = pd.merge(df,
                                          pd.DataFrame(fig_data[var_name][prod_name],
                                                       columns=['date',
                                                                '{}'.format(self.ifeature)]),
                                          on='date', how='inner')
                            df_col.append('var_{}'.format(ivar))
                        df.columns = df_col

                        # calculate data
                        print('{:>10d}'
                              '{:>40s}'.format(iplt, ylabel))
                        df['date'] = pd.to_datetime(df['date'], format='%Y-%m')

                        ivar = 0
                        df[ylabel] = df['var_{}'.format(ivar)]
                        for ioper in range(len(var_opers)):
                            var_oper = var_opers[ioper]
                            ivar = ioper + 1
                            if var_oper == '-':
                                df[ylabel] = df[ylabel] - df['var_{}'.format(ivar)]
                            if var_oper == '+':
                                df[ylabel] = df[ylabel] + df['var_{}'.format(ivar)]
                            if var_oper == '.':
                                # RMSE
                                y_rms = RMSE(df[ylabel], df['var_{}'.format(ivar)])
                        # print(df)

                        # plot data
                        # axes[i, j].bar(x=y['date'], height=y[ylabel], width=30)
                        axes[i, j].plot(df['date'], df[ylabel],
                                        color='black',
                                        linestyle='solid',
                                        linewidth=1,
                                        label='WB')
                        axes[i, j].plot(df['date'], df['var_{}'.format(ivar)],
                                        color='blue',
                                        linestyle='solid',
                                        linewidth=1,
                                        label='Q')
                        ax_leg = axes[i, j].legend(loc='upper right',
                                                  markerscale=0.5,
                                                  markerfirst=True,
                                                  fontsize=ax_legendsize,
                                                  labelspacing=0.2,
                                                  fancybox=False,
                                                  shadow=False,
                                                  framealpha=1.0, frameon=False)
                        for ax_leg_line in ax_leg.get_lines():
                            ax_leg_line.set_linewidth(0.5)
                        axes[i, j].set_title('RMSE {:0.2f}'.format(y_rms),
                                             fontsize=ax_titlesize)
                        axes[i, j].xaxis.set_major_locator(mdates.YearLocator())
                        axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                        axes[i, j].tick_params(axis='x', which='major', direction='in',
                                               pad=0.5, length=1,
                                               labelsize=ax_ticksize,
                                               labelrotation=45)
                        axes[i, j].tick_params(axis='y', which='major', direction='in',
                                               pad=0.5, length=1,
                                               labelsize=ax_ticksize,
                                               labelrotation=90)
                        axes[i, j].set_xlabel(xlabel,
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                        axes[i, j].set_ylabel(ylabel,
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                    else:
                        axes[i, j].remove()

            # fig.autofmt_xdate()
            fig.subplots_adjust(bottom=0.05, top=0.9,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.4)
            self.saveas(fig, name)
            self.close(fig)

        # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
        #               "potato", "wheat", "barley"]
        # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
        #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
        #
        # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
        #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
        #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
        #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
        #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
        #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
        #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
        #
        # fig, ax = plt.subplots()
        # im = ax.imshow(harvest)
        #
        # # We want to show all ticks...
        # ax.set_xticks(np.arange(len(farmers)))
        # ax.set_yticks(np.arange(len(vegetables)))
        # # ... and label them with the respective list entries
        # ax.set_xticklabels(farmers)
        # ax.set_yticklabels(vegetables)
        #
        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")
        #
        # # Loop over data dimensions and create text annotations.
        # for i in range(len(vegetables)):
        #     for j in range(len(farmers)):
        #         text = ax.text(j, i, harvest[i, j],
        #                        ha="center", va="center", color="w")
        #
        # ax.set_title("Harvest of local farmers (in tons/year)")
        # fig.tight_layout()
        # plt.show()
        # # plt.savefig()
        # # plt.clf()

    def plot_bar_wb(self, name):
        fig_conf = self.conf[name]
        # parse yaml
        fig_data = {}
        var_names = []
        var_opers = []
        tmp_names = fig_conf['data'].split('-')
        isub = 0
        for variable in tmp_names:
            if len(variable.split('+')) > 1:
                ipls = 0
                for sub_var in variable.split('+'):
                    if ipls > 0:
                        var_opers.append('+')
                    else:
                        var_opers.append('-')
                    var_names.append(sub_var)
                    fig_data[sub_var] = self.data[sub_var]
                    ipls += 1
            else:
                if isub > 0:
                    var_opers.append('-')
                var_names.append(variable)
                fig_data[variable] = self.data[variable]
            isub += 1
        # print(var_names, var_opers)

        fig_title = fig_conf['title']
        print(fig_title)

        fig_nplt = 1
        prod_names = []
        prod_nprod = []
        for ivar in range(len(var_names)):
            prod_names.append(list(fig_data[var_names[ivar]].keys()))
            prod_nprod.append([i for i in range(len(fig_data[var_names[ivar]].keys()))])
            fig_nplt *= np.prod(len(fig_data[var_names[ivar]].keys()))

        fig_nrow = int(np.floor(np.sqrt(fig_nplt)))
        fig_ncol = int(np.ceil(float(fig_nplt) / float(fig_nrow)))

        fig = fig_conf['obj']
        fig.suptitle(fig_title)
        axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)

        ax_titlesize = 6
        ax_ticksize = 4
        ax_labelsize = 4

        fig_comb = list(itertools.product(*prod_nprod))
        # print(len(fig_comb), prod_names)
        if len(fig_comb) > 0:
            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j
                    if iplt < fig_nplt:
                        prod_list = list(fig_comb[iplt])

                        # prepare data
                        ivar = 0
                        var_name = var_names[ivar]
                        prod_name = prod_names[ivar][prod_list[ivar]]

                        xlabel = 'date'
                        ylabel = '{}'.format(prod_name)

                        df_col = ['date']
                        df = pd.DataFrame(fig_data[var_name][prod_name],
                                          columns=['date',
                                                   '{}'.format(self.ifeature)])
                        df_col.append('var_{}'.format(ivar))
                        for ioper in range(len(var_opers)):
                            ivar = ioper + 1
                            var_oper = var_opers[ioper]
                            var_name = var_names[ivar]
                            prod_name = prod_names[ivar][prod_list[ivar]]

                            ylabel += '{}{}'.format(var_oper, prod_name)
                            df = pd.merge(df,
                                         pd.DataFrame(fig_data[var_name][prod_name],
                                                      columns=['date',
                                                               '{}'.format(self.ifeature)]),
                                         on='date', how='inner')
                            df_col.append('var_{}'.format(ivar))
                        df.columns = df_col

                        # calculate data
                        print('{:>10d}'
                              '{:>40s}'.format(iplt, ylabel))
                        df['date'] = pd.to_datetime(df['date'], format='%Y-%m')

                        ivar = 0
                        df[ylabel] = df['var_{}'.format(ivar)]
                        for ioper in range(len(var_opers)):
                            var_oper = var_opers[ioper]
                            ivar = ioper + 1
                            if var_oper == '-':
                                df[ylabel] = df[ylabel] - df['var_{}'.format(ivar)]
                            if var_oper == '+':
                                df[ylabel] = df[ylabel] + df['var_{}'.format(ivar)]
                        # print(df)

                        y_std = np.nanstd(df[ylabel])
                        y_avg = np.nanmean(df[ylabel])

                        # plot data
                        axes[i, j].bar(x=df['date'], height=df[ylabel], width=30)
                        axes[i, j].set_title('STD {:0.2f} '
                                             'AVG {:0.2f}'.format(y_std, y_avg),
                                             fontsize=ax_titlesize)
                        axes[i, j].xaxis.set_major_locator(mdates.YearLocator())
                        axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                        axes[i, j].tick_params(axis='x', which='major', direction='in',
                                               pad=0.5, length=1,
                                               labelsize=ax_ticksize,
                                               labelrotation=45)
                        axes[i, j].tick_params(axis='y', which='major', direction='in',
                                               pad=0.5, length=1,
                                               labelsize=ax_ticksize,
                                               labelrotation=90)
                        axes[i, j].set_xlabel(xlabel,
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                        axes[i, j].set_ylabel(ylabel,
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                    else:
                        axes[i, j].remove()

            # fig.autofmt_xdate()
            if fig_nplt < fig_nrow * fig_ncol:
                fig.text(x=0.8, y=0.05,
                         s='STD: Standard Deviation\nAVG: Mean',
                         fontsize=4,
                         horizontalalignment='left', verticalalignment='center')
            fig.subplots_adjust(bottom=0.05, top=0.9,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.4)
            self.saveas(fig, name)
            self.close(fig)

    def plot_scatter_prod(self, name):
        fig_conf = self.conf[name]
        # parse yaml
        fig_data = self.data[fig_conf['data']]

        prod_names = list(fig_data.keys())
        prod_nprod = len(prod_names)

        if prod_nprod > 1:
            fig_title = fig_conf['title']
            print(fig_title)

            fig_nplt = np.sum([i for i in range(prod_nprod)])
            fig_nrow = int(np.floor(np.sqrt(fig_nplt)))
            fig_ncol = int(np.ceil(float(fig_nplt) / float(fig_nrow)))

            fig = fig_conf['obj']
            fig.suptitle(fig_title)
            axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)

            ax_titlesize = 6
            ax_ticksize = 4
            ax_labelsize = 4

            fig_comb = []
            for i in range(prod_nprod):
                for j in range(prod_nprod):
                    if j > i:
                        fig_comb.append([i, j])

            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j
                    if iplt < fig_nplt:
                        print('{:>10d}'
                              '{:>20s}{:>20s}'.format(iplt,
                                                      prod_names[fig_comb[iplt][0]],
                                                      prod_names[fig_comb[iplt][1]]))

                        # prepare data
                        xlabel = prod_names[fig_comb[iplt][1]]
                        ylabel = prod_names[fig_comb[iplt][0]]

                        x = pd.DataFrame(fig_data[xlabel],
                                         columns=['date', '{}'.format(self.ifeature)])
                        y = pd.DataFrame(fig_data[ylabel],
                                         columns=['date', '{}'.format(self.ifeature)])

                        # calculate data
                        df = pd.merge(y, x, on='date', how='outer',
                                      suffixes=('_l', '_r'))
                        df.columns = ['data', 'y', 'x']
                        df['c'] = abs(df['y'] / df['x'] - 1.0)
                        # print(df)

                        # Pearson correlation coefficient
                        y_pcc = np.corrcoef(df['x'], df['y'])[0, 1]

                        # plot data
                        drange = [min(df['x'].min(), df['y'].min()),
                                  max(df['x'].max(), df['y'].max())]

                        axes[i, j].plot(drange, drange,
                                        color='black',
                                        linestyle='dashed',
                                        linewidth=1)

                        axes[i, j].scatter(x=df['x'], y=df['y'],
                                           c=df['c'], cmap='viridis',
                                           s=2.0, alpha=0.8)
                        axes[i, j].set_title('PCC {:0.2f}'.format(y_pcc),
                                             fontsize=ax_titlesize)
                        # axes[i, j].set_aspect(aspect='equal')
                        # print(axes[i, j].get_xticklabels())
                        axes[i, j].tick_params(axis='x', which='major', direction='in',
                                               pad=0.5, length=1,
                                               labelsize=ax_ticksize,
                                               labelrotation=0)
                        axes[i, j].tick_params(axis='y', which='major', direction='in',
                                               pad=0.5, length=1,
                                               labelsize=ax_ticksize,
                                               labelrotation=90)
                        axes[i, j].set_xlabel(xlabel,
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                        axes[i, j].set_ylabel(ylabel,
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                    else:
                        axes[i, j].remove()

            # fig.autofmt_xdate()
            if fig_nplt < fig_nrow * fig_ncol:
                fig.text(x=0.8, y=0.05,
                         s='PCC: Pearson correlation coefficient',
                         fontsize=4,
                         horizontalalignment='left', verticalalignment='center')
            fig.subplots_adjust(bottom=0.05, top=0.9,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.4)
            self.saveas(fig, name)
            self.close(fig)

    def saveas(self, fig, name):
        fig_ext = 'jpg'
        fig.savefig(os.path.join(self.workspace,
                                 '{}.{}'.format(name, fig_ext)),
                    format=fig_ext)
        fig_ext = 'pdf'
        fig.savefig(os.path.join(self.workspace,
                                 '{}.{}'.format(name, fig_ext)),
                    format=fig_ext)
        fig.clf()

    def close(self, fig):
        plt.close(fig)
