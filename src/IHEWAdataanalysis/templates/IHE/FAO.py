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

# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

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

try:
    from .topo import intersection
except ImportError:
    from IHEWAdataanalysis.topo import intersection


class Template(object):
    """This Base class

    Load base.yml file.

    if 0 < fig_naxe <= 10:
        ax_titlesize = 6
        ax_ticksize = 4
        ax_labelsize = 4
        ax_legendsize = 3
    if 10 < fig_naxe <= 20:
        ax_titlesize = 6
        ax_ticksize = 4
        ax_labelsize = 4
        ax_legendsize = 3
    if 20 < fig_naxe:
        ax_titlesize = 3
        ax_ticksize = 2
        ax_labelsize = 2
        ax_legendsize = 2


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
            'bar_prod',
            'bar_wb',
            'heatmap_wb',
            'line_prod',
            'line_wb',
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
        # print(self.conf.keys())

        data = self._data(self.__conf['path'])
        if len(data.keys()) > 0:
            self.data = data
        else:
            raise IHEClassInitError(template) from None
        # print(self.data.keys())

        print('\nFigure Start')
        self.workspace = os.path.join(self.__conf['path'], 'IHEWAdataanalysis', 'fig')
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

        print('>>>>>')
        for fig_name, fig_obj in self.conf.items():
            # print('{} {} "{}"'.format(fig_obj['obj'].number,
            #                           fig_name,
            #                           fig_obj['obj'].get_size_inches() *
            #                           fig_obj['obj'].dpi))
            ptype = fig_obj['ptype']
            if ptype in self.allow_ptypes:
                if ptype == 'bar_prod':
                    self.plot_bar_prod(fig_name)
                if ptype == 'bar_wb':
                    self.plot_bar_wb(fig_name)
                if ptype == 'heatmap_wb':
                    self.plot_heatmap_wb(fig_name)
                if ptype == 'line_prod':
                    self.plot_line_prod(fig_name)
                if ptype == 'line_wb':
                    self.plot_line_wb(fig_name)
                if ptype == 'scatter_prod':
                    self.plot_scatter_prod(fig_name)
            else:
                print('Warning "{}" not support.'.format(ptype))

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
                                    data[variable][prod_name] = \
                                        pd.read_csv(file,
                                                    index_col='date',
                                                    parse_dates=True)
                                    # print('{}'.format(
                                    #     data[variable][prod_name].describe()))
                        else:
                            data = {}

        return data

    # def create(self, conf) -> dict:
    #     objs = {}
    #
    #     for fig_name, fig_conf in conf.items():
    #         objs[fig_name] = fig_conf
    #         objs[fig_name]['obj'] = plt.figure(**fig_conf['figure'])
    #
    #     return objs

    def plot_bar_prod(self, name):
        fig_conf = self.conf[name]
        fig_title = fig_conf['title']
        print(fig_title)

        # parse yaml
        fig_data = self.data[fig_conf['data']]

        prod_names = list(fig_data.keys())
        prod_nprod = len(prod_names)

        if prod_nprod > 0:
            fig_nbar = prod_nprod

            # Yearly
            fig = plt.figure(**fig_conf['figure'])
            fig.subplots_adjust(bottom=0.15, top=0.9,
                                left=0.075, right=0.95)

            axes = fig.subplots(nrows=1, ncols=1, squeeze=False)
            ax_ticksize = 6
            ax_labelsize = 8
            xlabel = 'date'
            ax_bar_width = max(0.01, (1.0 / fig_nbar - 0.05))

            # fig.suptitle(fig_title)
            for i in range(fig_nbar):
                ax_yearly_xticks = []

                prod_name = prod_names[i]
                print('{:>10d}{:>20s}'.format(i, prod_name))

                # prepare data
                ylabel = '{}'.format(prod_name)
                df = pd.DataFrame(fig_data[ylabel],
                                  columns=['{}'.format(self.ifeature)])

                # calculate data
                df_yearly = df.groupby(df.index.year)
                df_yearly_sum = df_yearly.sum()
                for df_year, df_year_val in df_yearly_sum.iterrows():
                    # print(df_year.year)
                    ax_yearly_xticks.append(int(df_year))

                df_yearly_x = np.arange(0, len(ax_yearly_xticks), 1) - \
                    ax_bar_width * ((fig_nbar - 1.) / 2. - i)

                # plot data
                axes[0, 0].bar(x=df_yearly_x,
                               height=df_yearly_sum['{}'.format(self.ifeature)],
                               width=ax_bar_width,
                               label='{}'.format(prod_name))

            axes[0, 0].set_xticks(np.arange(0, len(ax_yearly_xticks), 1))
            axes[0, 0].set_xticklabels(ax_yearly_xticks)
            axes[0, 0].tick_params(axis='x', which='major', direction='in',
                                   pad=0.5, length=1,
                                   labelsize=ax_ticksize,
                                   labelrotation=0)
            axes[0, 0].tick_params(axis='y', which='major', direction='in',
                                   pad=0.5, length=1,
                                   labelsize=ax_ticksize,
                                   labelrotation=90)
            # axes[0, 0].set_xlabel(xlabel,
            #                       fontsize=ax_labelsize,
            #                       labelpad=1)
            axes[0, 0].set_ylabel(fig_title,
                                  fontsize=ax_labelsize,
                                  labelpad=1)
            axes[0, 0].legend()
            axes[0, 0].grid(True, which='major',
                            color='#999999', linewidth=1, linestyle='-', alpha=0.2)
            fig.subplots_adjust(bottom=0.05, top=0.9,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.4)
            self.saveas(fig, '{}_{}'.format(name, 'yearly'))
            self.close(fig)

            # Monthly
            fig = plt.figure(**fig_conf['figure'])

            axes = fig.subplots(nrows=1, ncols=1, squeeze=False)
            ax_bar_width = max(0.01, (1.0 / fig_nbar - 0.05))

            # fig.suptitle(fig_title)
            for i in range(fig_nbar):
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ax_hydrological_year = self.__conf['data']['hydrology']['year']
                ax_monthly_xticks = [months[i - 1] for i in ax_hydrological_year]

                prod_name = prod_names[i]
                print('{:>10d}{:>20s}'.format(i, prod_name))

                # prepare data
                ylabel = '{}'.format(prod_name)
                df = pd.DataFrame(fig_data[ylabel],
                                  columns=['{}'.format(self.ifeature)])

                # calculate data
                df_monthly_avg = df.groupby(df.index.month).mean()
                df_monthly_x = np.arange(0, len(ax_monthly_xticks), 1) - \
                    ax_bar_width * ((fig_nbar - 1.) / 2. - i)

                df_monthly_avg = df_monthly_avg.reindex(ax_hydrological_year)

                # plot data
                axes[0, 0].bar(x=df_monthly_x,
                               height=df_monthly_avg['{}'.format(self.ifeature)],
                               width=ax_bar_width,
                               label='{}'.format(prod_name))

            axes[0, 0].set_xticks(np.arange(0, len(ax_monthly_xticks), 1))
            axes[0, 0].set_xticklabels(ax_monthly_xticks)
            axes[0, 0].tick_params(axis='x', which='major', direction='in',
                                   pad=0.5, length=1,
                                   labelsize=ax_ticksize,
                                   labelrotation=0)
            axes[0, 0].tick_params(axis='y', which='major', direction='in',
                                   pad=0.5, length=1,
                                   labelsize=ax_ticksize,
                                   labelrotation=90)
            axes[0, 0].set_ylabel(fig_title,
                                  fontsize=ax_labelsize,
                                  labelpad=1)
            axes[0, 0].legend()
            axes[0, 0].grid(True, which='major',
                            color='#999999', linewidth=1, linestyle='-', alpha=0.2)
            fig.subplots_adjust(bottom=0.15, top=0.9,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.4)

            self.saveas(fig, '{}_{}'.format(name, 'monthly'))
            self.close(fig)

    def plot_bar_wb(self, name):
        fig_conf = self.conf[name]
        fig_title = fig_conf['title']
        print(fig_title)

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

        fig_naxe = 1
        prod_names = []
        prod_nprod = []
        for ivar in range(len(var_names)):
            prod_names.append(list(fig_data[var_names[ivar]].keys()))
            prod_nprod.append([i for i in range(len(fig_data[var_names[ivar]].keys()))])
            fig_naxe *= np.prod(len(fig_data[var_names[ivar]].keys()))

        fig_nrow = int(np.floor(np.sqrt(fig_naxe)))
        fig_ncol = int(np.ceil(float(fig_naxe) / float(fig_nrow)))
        fig_comb = list(itertools.product(*prod_nprod))
        # print(len(fig_comb), prod_names)

        fig = plt.figure(**fig_conf['figure'])
        if 0 < fig_nrow <= 2:
            fig.subplots_adjust(bottom=0.15, top=0.8,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.4)
        if 2 < fig_nrow <= 5:
            fig.set_size_inches(6.4, 4.8, forward=True)
            fig.subplots_adjust(bottom=0.1, top=0.85,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.45)
        if 5 < fig_nrow:
            fig.set_size_inches(6.4, 4.8, forward=True)
            fig.subplots_adjust(bottom=0.05, top=0.95,
                                left=0.05, right=0.9)

        axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)
        if 0 < fig_naxe <= 10:
            ax_titlesize = 6
            ax_ticksize = 4
            ax_labelsize = 4
            ax_legendsize = 3
        if 10 < fig_naxe <= 20:
            ax_titlesize = 6
            ax_ticksize = 4
            ax_labelsize = 4
            ax_legendsize = 3
        if 20 < fig_naxe:
            ax_titlesize = 3
            ax_ticksize = 2
            ax_labelsize = 2
            ax_legendsize = 2
        ax_ylim = [np.inf, -np.inf]

        # fig.suptitle(fig_title)
        if len(fig_comb) > 0:
            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j
                    if iplt < fig_naxe:
                        prod_list = list(fig_comb[iplt])

                        # prepare data
                        ivar = 0
                        var_name = var_names[ivar]
                        prod_name = prod_names[ivar][prod_list[ivar]]

                        xlabel = 'date'
                        ylabel = '{}'.format(prod_name)

                        df_col = []
                        df = pd.DataFrame(fig_data[var_name][prod_name],
                                          columns=['{}'.format(self.ifeature)])
                        df_col.append('var_{}'.format(ivar))
                        for ioper in range(len(var_opers)):
                            ivar = ioper + 1
                            var_oper = var_opers[ioper]
                            var_name = var_names[ivar]
                            prod_name = prod_names[ivar][prod_list[ivar]]

                            ylabel += '{}{}'.format(var_oper, prod_name)
                            df = pd.merge(df,
                                          pd.DataFrame(fig_data[var_name][prod_name],
                                                       columns=['{}'.format(
                                                           self.ifeature)]),
                                          left_index=True, right_index=True,
                                          how='inner')
                            df_col.append('var_{}'.format(ivar))
                        df.columns = df_col

                        # calculate data
                        print('{:>10d}'
                              '{:>40s}'.format(iplt, ylabel))

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
                        ax_ylim[0] = min(ax_ylim[0],
                                         min(np.min(df[ylabel]),
                                             np.min(df['var_{}'.format(ivar)])))
                        ax_ylim[1] = max(ax_ylim[1],
                                         max(np.max(df[ylabel]),
                                             np.max(df['var_{}'.format(ivar)])))

                        axes[i, j].plot([df.index[0], df.index[-1]],
                                        [0., 0.],
                                        color='gray',
                                        linestyle='solid',
                                        linewidth=0.5)

                        axes[i, j].bar(x=df.index, height=df[ylabel], width=30,
                                       color='black')
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
                        # if fig_naxe < 6:
                        #     axes[i, j].grid(True)
                    else:
                        axes[i, j].remove()

            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j
                    if iplt < fig_naxe:
                        axes[i, j].set_ylim(tuple(ax_ylim))

            if fig_nrow > 4:
                fig.autofmt_xdate()

            self.saveas(fig, name)
            self.close(fig)

    def heatmap(self, data, row_labels, col_labels, ax=None, **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(),
                 ha="center",
                 rotation=0,
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(),
                 ha="center",
                 rotation=90,
                 rotation_mode="anchor")

        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.set_frame_on(False)

        ax.grid(which="major", b=False)
        ax.tick_params(which="major",
                       bottom=False, left=False, top=False, right=False,
                       labeltop=True, labelbottom=False)

        # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.tick_params(which="minor",
                       bottom=False, left=False, top=False, right=False,
                       labeltop=False, labelbottom=False)

        return im

    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def plot_heatmap_wb(self, name):
        fig_conf = self.conf[name]
        fig_title = fig_conf['title']
        print(fig_title)

        # parse yaml
        fig_data = {}
        var_names = []
        var_opers = []
        tmp_names_root = fig_conf['data'].split('.')
        iagn = 0
        for variable_root in tmp_names_root:
            # if len(variable_root.split('_')) > 1:
            #     tmp_names_subroot = variable_root.split('_')
            #     isub = 0
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

        ax_legends = ['PCC', 'R2', 'RMSE']
        fig_nscr = len(ax_legends)
        fig_pix_naxe = 1
        fig_naxe = 1
        prod_names = []
        prod_nprod = []
        for ivar in range(len(var_names)):
            prod_names.append(list(fig_data[var_names[ivar]].keys()))
            prod_nprod.append([i for i in range(len(fig_data[var_names[ivar]].keys()))])

            if 'PCP' in var_names[ivar]:
                ax_xticks = list(fig_data[var_names[ivar]].keys())
                fig_pix_ncol = int(len(ax_xticks))
            if 'ET' in var_names[ivar]:
                ax_yticks = list(fig_data[var_names[ivar]].keys())
                fig_pix_nrow = int(len(ax_yticks))
            if 'dS' == var_names[ivar]:
                ax_titles = list(fig_data[var_names[ivar]].keys())
                fig_nrow = int(len(ax_titles))
            # if var_names[ivar] == 'Q':
            #     fig_ncol = int(len(fig_data[var_names[ivar]].keys()))
        # fig_ncol = int(1)
        fig_ncol = fig_nscr
        fig_pix_naxe = int(fig_pix_nrow * fig_pix_ncol)
        fig_naxe = int(fig_nrow * fig_ncol)
        fig_pix_comb = list(itertools.product(*prod_nprod))
        # print(len(fig_comb), prod_names)

        ax_ylim = [[np.inf, -np.inf],
                   [np.inf, -np.inf],
                   [np.inf, -np.inf]]
        if len(fig_pix_comb) > 0:
            y = np.empty(((fig_pix_nrow * fig_pix_ncol), fig_naxe, fig_nscr))
            y[:] = np.nan
            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j

                    for ii in range(fig_pix_nrow):
                        for jj in range(fig_pix_ncol):
                            ipix = ii * fig_pix_ncol + jj
                            if ipix < fig_pix_naxe:
                                prod_list = list(fig_pix_comb[ipix])

                                # prepare data
                                ivar = 0
                                var_name = var_names[ivar]
                                prod_name = prod_names[ivar][prod_list[ivar]]

                                xlabel = 'date'
                                ylabel = '{}'.format(prod_name)

                                df_col = []
                                df = pd.DataFrame(fig_data[var_name][prod_name],
                                                  columns=['{}'.format(self.ifeature)])
                                df_col.append('var_{}'.format(ivar))
                                for ioper in range(len(var_opers)):
                                    ivar = ioper + 1
                                    var_oper = var_opers[ioper]
                                    var_name = var_names[ivar]
                                    prod_name = prod_names[ivar][prod_list[ivar]]

                                    ylabel += '{}{}'.format(var_oper, prod_name)
                                    df = pd.merge(df,
                                                  pd.DataFrame(
                                                      fig_data[var_name][prod_name],
                                                      columns=['{}'.format(
                                                          self.ifeature)]),
                                                  left_index=True, right_index=True,
                                                  how='inner')
                                    df_col.append('var_{}'.format(ivar))
                                df.columns = df_col

                                # calculate data
                                print('{:>10d}'
                                      '{:>40s}'.format(ipix, ylabel))

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
                                        # PCC, Pearson correlation coefficient
                                        y[ipix, iplt, 0] = np.corrcoef(
                                            df[ylabel], df['var_{}'.format(ivar)])[0, 1]
                                        ax_ylim[0] = [min(ax_ylim[0][0],
                                                          np.min(y[ipix, iplt, 0])),
                                                      max(ax_ylim[0][1],
                                                          np.max(y[ipix, iplt, 0]))]
                                        # R2
                                        y[ipix, iplt, 1] = r2_score(
                                            df[ylabel], df['var_{}'.format(ivar)])
                                        ax_ylim[1] = [min(ax_ylim[1][0],
                                                          np.min(y[ipix, iplt, 1])),
                                                      max(ax_ylim[1][1],
                                                          np.max(y[ipix, iplt, 1]))]
                                        # RMSE
                                        y[ipix, iplt, 2] = RMSE(
                                            df[ylabel], df['var_{}'.format(ivar)])
                                        ax_ylim[2] = [min(ax_ylim[2][0],
                                                          np.min(y[ipix, iplt, 2])),
                                                      max(ax_ylim[2][1],
                                                          np.max(y[ipix, iplt, 2]))]
                                        # print(ipix, y[ipix, iplt, 2])
                                # print(df)

            # plot combine
            fig = plt.figure(**fig_conf['figure'])
            fig.set_size_inches(6.4, 4.8 + 4.8 * (fig_nrow - 1.0) / 2.0, forward=True)
            fig.subplots_adjust(bottom=0.15, top=0.95,
                                left=0.1, right=0.9,)

            axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)
            ax_ticksize = max(9 - fig_nrow, 1)
            ax_labelsize = max(11 - fig_nrow, 1)
            ax_textsize = max(7 - fig_nrow, 1)
            ax_cbarsize = max(7 - fig_nrow, 1)
            ax_cbarcmap = ['Blues', 'Blues', 'Blues']
            # ax_cbarcmap = ['RdYlBu_r', 'RdYlGn_r', 'Reds_r']

            # fig.suptitle(fig_title)
            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j

                    tmp_y = np.empty((fig_pix_nrow, fig_pix_ncol))
                    tmp_y[:] = np.nan
                    for ii in range(fig_pix_nrow):
                        for jj in range(fig_pix_ncol):
                            ipix = jj * fig_pix_nrow + ii
                            tmp_y[ii, jj] = y[ipix, iplt, j]
                    im = self.heatmap(data=tmp_y,
                                      row_labels=ax_yticks,
                                      col_labels=ax_xticks,
                                      ax=axes[i, j],
                                      cmap=ax_cbarcmap[j],
                                      vmin=ax_ylim[j][0],
                                      vmax=ax_ylim[j][1]
                                      )
                    texts = self.annotate_heatmap(im,
                                                  valfmt="{x:.2f}",
                                                  fontsize=ax_textsize)

                    axes[i, j].tick_params(axis='x', labelsize=ax_ticksize)
                    axes[i, j].tick_params(axis='y', labelsize=ax_ticksize)
                    axes[i, j].set_ylabel(ax_titles[i],
                                          fontsize=ax_labelsize,
                                          labelpad=1)
                    axes[i, j].yaxis.set_label_position("right")

                    # if i == 0:
                    #     axes[i, j].set_title('{}'.format(ax_legends[j]),
                    #                          fontsize=ax_labelsize)

                    # Create colorbar
                    if i == fig_nrow - 1:
                        ax_pos = axes[i, j].get_position()

                        ax_cb_l = ax_pos.x0
                        ax_cb_b = max(0.05, ax_pos.y0 - 0.1)
                        ax_cb_w = ax_pos.x1 - ax_pos.x0
                        ax_cb_h = 0.02
                        # print('cbar {:0.4f} {:0.4f} {:0.4f} {:0.4f}'.format(
                        #     ax_cb_l, ax_cb_b, ax_cb_w, ax_cb_h))

                        cax = fig.add_axes([ax_cb_l, ax_cb_b, ax_cb_w, ax_cb_h])
                        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), cax=cax,
                                            orientation="horizontal")
                        cbar.ax.set_title('{}'.format(ax_legends[j]),
                                          fontsize=ax_labelsize)
                        cbar.ax.tick_params(axis='x', which='major', direction='in',
                                            pad=0.5, length=1,
                                            labelsize=ax_cbarsize,
                                            labelrotation=0)

            self.saveas(fig, '{}'.format(name))
            self.close(fig)

            # plot singel
            fig_ncol = int(1)
            for k in range(fig_nscr):
                fig = plt.figure(**fig_conf['figure'])
                fig.set_size_inches(4.8, 6.4, forward=True)
                fig.subplots_adjust(bottom=0.1, top=0.9, left=0.2, right=0.75)

                axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)
                ax_ticksize = max(9 - fig_nrow, 1)
                ax_labelsize = max(11 - fig_nrow, 1)
                ax_textsize = max(7 - fig_nrow, 1)
                ax_cbarsize = max(7 - fig_nrow, 1)
                ax_cbarcmap = ['Blues', 'Blues', 'Blues']
                # ax_cbarcmap = ['RdYlBu_r', 'RdYlGn_r', 'Reds_r']

                # fig.suptitle(fig_title)
                for i in range(fig_nrow):
                    for j in range(fig_ncol):
                        iplt = i * fig_ncol + j

                        tmp_y = np.empty((fig_pix_nrow, fig_pix_ncol))
                        tmp_y[:] = np.nan
                        for ii in range(fig_pix_nrow):
                            for jj in range(fig_pix_ncol):
                                ipix = jj * fig_pix_nrow + ii
                                tmp_y[ii, jj] = y[ipix, iplt, k]
                        im = self.heatmap(data=tmp_y,
                                          row_labels=ax_yticks,
                                          col_labels=ax_xticks,
                                          ax=axes[i, j],
                                          cmap=ax_cbarcmap[j],
                                          vmin=ax_ylim[k][0],
                                          vmax=ax_ylim[k][1]
                                          )
                        texts = self.annotate_heatmap(im,
                                                      valfmt="{x:.2f}",
                                                      fontsize=ax_textsize)

                        axes[i, j].tick_params(axis='x', labelsize=ax_ticksize)
                        axes[i, j].tick_params(axis='y', labelsize=ax_ticksize)
                        axes[i, j].set_ylabel(ax_titles[i],
                                              fontsize=ax_labelsize,
                                              labelpad=1)
                        axes[i, j].yaxis.set_label_position("right")

                        if j == fig_ncol - 1:
                            ax_pos = axes[i, j].get_position()

                # Create colorbar
                cax = fig.add_axes([ax_pos.x1 + 0.1, 0.2, 0.02, 0.5])
                cbar = plt.colorbar(im, ax=axes.ravel().tolist(), cax=cax)
                cbar.ax.set_title('{}'.format(ax_legends[k]))
                cbar.ax.tick_params(labelsize=ax_textsize)

                self.saveas(fig, '{}_{}'.format(name, ax_legends[k]))
                self.close(fig)

    def plot_line_prod(self, name):
        fig_conf = self.conf[name]
        fig_title = fig_conf['title']
        print(fig_title)

        # parse yaml
        fig_data = self.data[fig_conf['data']]

        prod_names = list(fig_data.keys())
        prod_nprod = len(prod_names)

        if prod_nprod > 0:
            fig_nbar = prod_nprod

            fig = plt.figure(**fig_conf['figure'])
            fig.subplots_adjust(bottom=0.15, top=0.9,
                                left=0.075, right=0.95)

            axes = fig.subplots(nrows=1, ncols=1, squeeze=False)
            ax_ticksize = 6
            ax_labelsize = 8
            xlabel = 'date'

            # fig.suptitle(fig_title)
            for i in range(fig_nbar):
                prod_name = prod_names[i]
                print('{:>10d}{:>20s}'.format(i, prod_name))

                # prepare data
                xlabel = 'date'
                ylabel = '{}'.format(prod_name)

                # x = pd.DataFrame(fig_data[xlabel],
                #                  columns=['{}'.format(self.ifeature)])
                df = pd.DataFrame(fig_data[ylabel],
                                  columns=['{}'.format(self.ifeature)])

                # calculate data
                # print(df)

                # plot data
                axes[0, 0].plot(df.index, df['{}'.format(self.ifeature)],
                                linestyle='solid',
                                linewidth=1,
                                label='{}'.format(prod_name))
                axes[0, 0].xaxis.set_major_locator(mdates.YearLocator())
                axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                axes[0, 0].tick_params(axis='x', which='major', direction='in',
                                       pad=0.5, length=1,
                                       labelsize=ax_ticksize,
                                       labelrotation=45)
                axes[0, 0].tick_params(axis='y', which='major', direction='in',
                                       pad=0.5, length=1,
                                       labelsize=ax_ticksize,
                                       labelrotation=90)
                # axes[0, 0].set_xlabel(xlabel,
                #                       fontsize=ax_labelsize,
                #                       labelpad=1)
                axes[0, 0].set_ylabel(fig_title,
                                      fontsize=ax_labelsize,
                                      labelpad=1)

            axes[0, 0].legend()
            axes[0, 0].grid(True, which='major',
                            color='#999999', linewidth=1, linestyle='-', alpha=0.2)

            self.saveas(fig, name)
            self.close(fig)

    def plot_line_wb(self, name):
        fig_conf = self.conf[name]
        fig_title = fig_conf['title']
        print(fig_title)

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

        fig_naxe = 1
        prod_names = []
        prod_nprod = []
        for ivar in range(len(var_names)):
            prod_names.append(list(fig_data[var_names[ivar]].keys()))
            prod_nprod.append([i for i in range(len(fig_data[var_names[ivar]].keys()))])
            fig_naxe *= np.prod(len(fig_data[var_names[ivar]].keys()))

        fig_nrow = int(np.floor(np.sqrt(fig_naxe)))
        fig_ncol = int(np.ceil(float(fig_naxe) / float(fig_nrow)))
        fig_comb = list(itertools.product(*prod_nprod))
        # print(len(fig_comb), prod_names)

        fig = plt.figure(**fig_conf['figure'])
        if 0 < fig_nrow <= 2:
            fig.subplots_adjust(bottom=0.15, top=0.8,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.4)
        if 2 < fig_nrow <= 5:
            fig.set_size_inches(6.4, 4.8, forward=True)
            fig.subplots_adjust(bottom=0.1, top=0.85,
                                left=0.075, right=0.95,
                                wspace=0.2, hspace=0.45)
        if 5 < fig_nrow:
            fig.set_size_inches(6.4, 4.8, forward=True)
            fig.subplots_adjust(bottom=0.05, top=0.95,
                                left=0.05, right=0.9)

        axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)
        if 0 < fig_naxe <= 10:
            ax_titlesize = 6
            ax_ticksize = 4
            ax_labelsize = 4
            ax_legendsize = 3
        if 10 < fig_naxe <= 20:
            ax_titlesize = 6
            ax_ticksize = 4
            ax_labelsize = 4
            ax_legendsize = 3
        if 20 < fig_naxe:
            ax_titlesize = 3
            ax_ticksize = 2
            ax_labelsize = 2
            ax_legendsize = 2
        ax_ylim = [np.inf, -np.inf]

        # fig.suptitle(fig_title)
        if len(fig_comb) > 0:
            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j
                    if iplt < fig_naxe:
                        prod_list = list(fig_comb[iplt])

                        # prepare data
                        ivar = 0
                        var_name = var_names[ivar]
                        prod_name = prod_names[ivar][prod_list[ivar]]

                        xlabel = 'date'
                        ylabel = '{}'.format(prod_name)

                        df_col = []
                        df = pd.DataFrame(fig_data[var_name][prod_name],
                                          columns=['{}'.format(self.ifeature)])
                        df_col.append('var_{}'.format(ivar))
                        for ioper in range(len(var_opers)):
                            ivar = ioper + 1
                            var_oper = var_opers[ioper]
                            var_name = var_names[ivar]
                            prod_name = prod_names[ivar][prod_list[ivar]]

                            ylabel += '{}{}'.format(var_oper, prod_name)
                            df = pd.merge(df,
                                          pd.DataFrame(fig_data[var_name][prod_name],
                                                       columns=['{}'.format(
                                                           self.ifeature)]),
                                          left_index=True, right_index=True,
                                          how='inner')
                            df_col.append('var_{}'.format(ivar))
                        df.columns = df_col

                        # calculate data
                        print('{:>10d}'
                              '{:>40s}'.format(iplt, ylabel))

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
                                # PCC, Pearson correlation coefficient
                                y_pcc = np.corrcoef(
                                    df[ylabel], df['var_{}'.format(ivar)])[0, 1]
                                # R2
                                y_r2 = r2_score(
                                    df[ylabel], df['var_{}'.format(ivar)])
                                # RMSE
                                y_rms = RMSE(
                                    df[ylabel], df['var_{}'.format(ivar)])
                        # print(df)

                        # plot data
                        ax_ylim[0] = min(ax_ylim[0],
                                         min(np.min(df[ylabel]),
                                             np.min(df['var_{}'.format(ivar)])))
                        ax_ylim[1] = max(ax_ylim[1],
                                         max(np.max(df[ylabel]),
                                             np.max(df['var_{}'.format(ivar)])))

                        # axes[i, j].bar(x=y['date'], height=y[ylabel], width=30)
                        axes[i, j].plot([df.index[0], df.index[-1]],
                                        [0., 0.],
                                        color='gray',
                                        linestyle='solid',
                                        linewidth=0.5)

                        axes[i, j].plot(df.index, df[ylabel],
                                        color='black',
                                        linestyle='solid',
                                        linewidth=1,
                                        label='WB')

                        # x = mdates.date2num(df.index)
                        # y = df[ylabel]
                        # ax_cmap = ListedColormap(['r', 'k'])
                        # ax_norm = BoundaryNorm([-1, 0, 1], ax_cmap.N)
                        # ax_points = np.array([x, y]).T.reshape(-1, 1, 2)
                        # ax_segments = np.concatenate([ax_points[:-1], ax_points[1:]], axis=1)
                        # ax_lc = LineCollection(ax_segments, cmap=ax_cmap, norm=ax_norm)
                        # ax_lc.set_array(y)
                        # ax_lc.set_linewidth(1)
                        # axes[i, j].add_collection(ax_lc)

                        axes[i, j].plot(df.index, df['var_{}'.format(ivar)],
                                        color='blue',
                                        linestyle='solid',
                                        linewidth=1,
                                        label='Q')

                        x = np.reshape(mdates.date2num(df.index), (-1, ))
                        y = np.reshape(df[ylabel].values, (-1, ))
                        int_x, int_y = intersection(x, y,
                                                    x, np.zeros(y.shape)-0.00001)
                        x = x.reshape(-1)
                        y = y.reshape(-1)
                        int_x = int_x.reshape(-1)
                        int_y = int_y.reshape(-1)
                        ii = np.searchsorted(x, int_x)
                        x = np.insert(x, ii, int_x)
                        y = np.insert(y, ii, int_y)
                        # for iint in range(int_x.shape[1]):
                        #     iloc = x.searchsorted(int_x[iint])
                        #     x = np.insert(x, iloc, int_x[iint])
                        #     y = np.insert(y, iloc, int_y[iint])

                        # x = np.arange(tmp_x[0], tmp_x[-1], 0.1)
                        # y = np.interp(x, tmp_x, tmp_y)
                        axes[i, j].plot(x,
                                        np.where(y > 0, np.nan, y),
                                        color='red',
                                        linestyle='solid',
                                        linewidth=1)
                        # axes[i, j].fill_between(df['date'],
                        #                         np.where(df[ylabel] > 0,
                        #                                  np.nan,
                        #                                  df[ylabel]),
                        #                         0,
                        #                         color='red')

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

                        axes[i, j].set_title('PCC {:0.2f} '
                                             'R2 {:0.2f} '
                                             'RMSE {:0.2f}'.format(y_pcc, y_r2, y_rms),
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
                        # if fig_naxe < 6:
                        #     axes[i, j].grid(True)
                    else:
                        axes[i, j].remove()

                for i in range(fig_nrow):
                    for j in range(fig_ncol):
                        iplt = i * fig_ncol + j
                        if iplt < fig_naxe:
                            axes[i, j].set_ylim(tuple(ax_ylim))
            if fig_nrow > 4:
                fig.autofmt_xdate()

            self.saveas(fig, name)
            self.close(fig)

    def plot_scatter_prod(self, name):
        fig_conf = self.conf[name]
        fig_title = fig_conf['title']
        print(fig_title)

        # parse yaml
        fig_data = self.data[fig_conf['data']]

        prod_names = list(fig_data.keys())
        prod_nprod = len(prod_names)

        if prod_nprod > 1:
            fig_naxe = np.sum([i for i in range(prod_nprod)])
            fig_nrow = int(np.floor(np.sqrt(fig_naxe)))
            fig_ncol = int(np.ceil(float(fig_naxe) / float(fig_nrow)))
            fig_comb = []
            for i in range(prod_nprod):
                for j in range(prod_nprod):
                    if j > i:
                        fig_comb.append([i, j])

            fig = plt.figure(**fig_conf['figure'])
            if fig_ncol == 1:
                fig.set_size_inches(4.8, 4.8, forward=True)
            if 0 < fig_nrow <= 2:
                fig.subplots_adjust(bottom=0.15, top=0.8,
                                    left=0.075, right=0.8,
                                    wspace=0.2, hspace=0.4)
            if 2 < fig_nrow <= 5:
                fig.set_size_inches(6.4, 4.8, forward=True)
                fig.subplots_adjust(bottom=0.1, top=0.85,
                                    left=0.075, right=0.8,
                                    wspace=0.2, hspace=0.45)
            if 5 < fig_nrow:
                fig.set_size_inches(6.4, 4.8, forward=True)
                fig.subplots_adjust(bottom=0.05, top=0.95,
                                    left=0.05, right=0.8)

            axes = fig.subplots(nrows=fig_nrow, ncols=fig_ncol, squeeze=False)
            ax_titlesize = 6
            ax_ticksize = 4
            ax_labelsize = 4

            # fig.suptitle(fig_title)
            for i in range(fig_nrow):
                for j in range(fig_ncol):
                    iplt = i * fig_ncol + j
                    if iplt < fig_naxe:
                        print('{:>10d}'
                              '{:>20s}{:>20s}'.format(iplt,
                                                      prod_names[fig_comb[iplt][0]],
                                                      prod_names[fig_comb[iplt][1]]))

                        # prepare data
                        xlabel = prod_names[fig_comb[iplt][1]]
                        ylabel = prod_names[fig_comb[iplt][0]]

                        # x = pd.DataFrame(fig_data[xlabel],
                        #                  columns=['date', '{}'.format(self.ifeature)])
                        # y = pd.DataFrame(fig_data[ylabel],
                        #                  columns=['date', '{}'.format(self.ifeature)])

                        # calculate data
                        df = pd.merge(fig_data[ylabel]['{}'.format(self.ifeature)],
                                      fig_data[xlabel]['{}'.format(self.ifeature)],
                                      left_index=True, right_index=True,
                                      how='outer',
                                      suffixes=('_l', '_r'))
                        df.columns = ['y', 'x']
                        xy = np.vstack([df['x'], df['y']])
                        df['c'] = gaussian_kde(xy)(xy)
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

                        im = axes[i, j].scatter(x=df['x'], y=df['y'],
                                                c=df['c'], cmap='jet',
                                                s=3.0, alpha=0.8)
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
                        if j == fig_ncol - 1:
                            ax_pos = axes[i, j].get_position()
                    else:
                        axes[i, j].remove()

            # Create colorbar
            ax_cb_l = min(ax_pos.x1 + 0.05, 0.9)
            ax_cb_b = 0.2
            ax_cb_w = 0.02
            ax_cb_h = 0.6
            # print('cbar {:0.4f} {:0.4f} {:0.4f} {:0.4f}'.format(
            #     ax_cb_l, ax_cb_b, ax_cb_w, ax_cb_h))

            cax = fig.add_axes([ax_cb_l, ax_cb_b, ax_cb_w, ax_cb_h])
            cbar = plt.colorbar(im, ax=axes.ravel().tolist(), cax=cax,
                                ticks=[df['c'].min(), df['c'].max()])
            cbar.ax.set_yticklabels(['Low', 'High'])
            cbar.ax.set_ylabel('Density', fontsize=ax_labelsize)
            cbar.ax.tick_params(axis='y', which='major', direction='in',
                                pad=0.5, length=1,
                                labelsize=ax_ticksize,
                                labelrotation=0)

            if fig_nrow > 4:
                fig.autofmt_xdate()

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
