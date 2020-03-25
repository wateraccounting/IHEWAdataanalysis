# -*- coding: utf-8 -*-
import inspect
import os
from datetime import datetime, date
import yaml

import matplotlib
matplotlib.use('Agg')  # Not to use X server. For TravisCI.
import matplotlib.pyplot as plt

import numpy as np
import quantities as pq

from pylatex import Package, Document, Command, NoEscape, \
    PageStyle, Head, Foot, \
    Section, Subsection, NewPage, NewLine, LineBreak, \
    Itemize, \
    Label, Ref, \
    LongTabu, LongTable, MultiColumn, MultiRow, Table, Tabular, \
    TikZ, Axis, Plot, Figure, Alignat, \
    Math, Matrix, VectorName, Quantity

from pylatex.utils import italic, make_temp_dir, rm_temp_dir

try:
    # IHEClassInitError, IHEStringError, IHETypeError, IHEKeyError, IHEFileError
    from .exception import IHEClassInitError
except ImportError:
    from IHEWAreport.exception import IHEClassInitError


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

        doc = self.create()
        if doc is not None:
            self.doc = doc

            print('\nLaTex Start')
            print('Create temp dir:', make_temp_dir())

            print('>>>>>')
            # doc Cover
            self.write_cover_page('CoverPage')
            self.write_title_page('FirstPage')

            # doc TOC
            self.write_toc_page('TOCPage')

            # doc Preamble style
            self.set_page('PreambleHeader', 'roman')

            # doc LOF, LOT
            self.write_lof_page('LOFPage')
            self.write_lot_page('LOTPage')

            # doc Preamble
            self.write_acknowledgment_page('AcknowledgementPage')
            self.write_abbreviation_page('AbbreviationPage')
            self.write_summary_page('SummaryPage')

            # doc Contents style
            self.set_page('SectionHeader', 'arabic')
            # doc Contents
            self.write_section_page('SectionPage')
            self.write_test_page('TestPage')

            # doc Appendix
            self.write_reference_page('ReferencePage')
            self.write_annex_page('AnnexPage')
            print('<<<<<\n')

            # doc Appendix
            self.saveas()
            self.close()
            print('\nLaTex End')

            print('Remove temp dir')
            rm_temp_dir()

    def _conf(self, path, template) -> dict:
        data = {}

        file_conf = os.path.join(path, template)
        with open(file_conf) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)

        return data

    def create(self) -> object:
        opt_geometry = self.data['layout']
        doc = Document(geometry_options=opt_geometry)
        # Header and footer
        doc.packages.append(Package('fancyhdr'))
        # Text align
        doc.packages.append(Package('ragged2e'))
        # Hyper link
        doc.packages.append(Package('hyperref'))
        # Reference
        # sorting=ynt
        #   nty—sorts entries by name, title, year;
        #   nyt—sorts entries by name, year, title;
        #   nyvt—sorts entries by name, year, volume, title;
        #   anyt—sorts entries by alphabetic label, name, year, title;
        #   anyvt—sorts entries by alphabetic label, name, year, volume, title;
        #   ynt—sorts entries by year, name, title;
        #   ydnt—sorts entries by year (descending order), name, title;
        #   none—no sorting. Entries appear in the order they appear in the text.
        doc.packages.append(Package('biblatex',
                                    options=[
                                        'style=authoryear',
                                        'sorting=ynt'
                                    ]))
        #                                 'backend=bibtex',
        # doc.packages.append(Package('natbib'))

        return doc

    def set_page(self, sname, ntype):
        opt_header = self.data['page']['header']
        if 'header' in self.__conf['data']['page'].keys():
            if self.__conf['data']['page']['header'] is not None:
                opt_header = self.__conf['data']['page']['header']

        opt_footer = self.data['page']['footer']
        if 'footer' in self.__conf['data']['page'].keys():
            if self.__conf['data']['page']['footer'] is not None:
                opt_footer = self.__conf['data']['page']['footer']

        doc_header = PageStyle(sname)

        with doc_header.create(Head('L')):
            self.page_header(doc_header, opt_header['left'])
        with doc_header.create(Head('C')):
            self.page_header(doc_header, opt_header['center'])
        with doc_header.create(Head('R')):
            self.page_header(doc_header, opt_header['right'])

        with doc_header.create(Foot('L')):
            self.page_header(doc_header, opt_footer['left'])
        with doc_header.create(Foot('C')):
            self.page_header(doc_header, opt_footer['center'])
        with doc_header.create(Foot('R')):
            self.page_header(doc_header, opt_footer['right'])

        self.doc.append(Command('clearpage'))

        self.doc.append(Command('pagenumbering', ntype))
        # self.doc.append(NoEscape(r'\setcounter{page}{1}'))
        self.doc.append(Command('setcounter', ['page', '1']))

        self.doc.preamble.append(doc_header)
        self.doc.change_document_style(sname)

    def page_header(self, obj, txt):
        if txt == 'page_number':
            obj.append(Command('thepage'))
        elif txt == 'simple_page_number':
            from pylatex import simple_page_number
            obj.append(simple_page_number())
        else:
            obj.append(txt)

    def insert_plot(self, obj_sec, name, caption, width, *args, **kwargs):
        # with obj_sec.create(Figure(position='htbp')) as plot:
        with obj_sec.create(Figure(position='h!')) as plot:
            plot.add_plot(width=NoEscape(width), *args, **kwargs)
            if isinstance(caption, str):
                if len(caption) > 0:
                    plot.add_caption('{}'.format(caption))
            plot.append(Label('figure:{}'.format(name)))
            # obj_sec.append(plot)

    def insert_image(self, obj_sec, name, caption, width, *args, **kwargs):
        pass

    def insert_abbre(self, obj_sec, width, data):
        tab_style = '{}'.format(' '.join(['X[l]' for i in range(2)]))
        with obj_sec.create(LongTabu(tab_style)) as table:
            table.end_table_header()
            table.end_table_footer()
            table.end_table_last_footer()

            for key, val in data.items():
                table.add_row([key, val])

    def insert_table(self, obj_sec, name, caption, width, header, data):
        # tab_style = '|{}|'.format('|'.join(['c' for i in range(data.shape[1])]))
        tab_style = '|{}|'.format('|'.join(['l' for i in range(data.shape[1])]))

        with obj_sec.create(LongTable(tab_style)) as table:
            if isinstance(caption, str):
                if len(caption) > 0:
                    table.append(Command('caption',
                                         options=[],
                                         arguments=['{}'.format(caption)]))
            table.append(NoEscape(r'\label{%s}\\' % 'table:{}'.format(name)))

            table.add_hline()
            table.add_row(header)
            table.add_hline()
            table.end_table_header()

            # table.add_hline()
            # table.add_row((MultiColumn(3,
            #                            align='r',
            #                            data='Continued on Next Page'),))
            table.add_hline()
            table.end_table_footer()

            # table.add_hline()
            # table.add_row((MultiColumn(3,
            #                            align='r',
            #                            data='Not Continued on Next Page'),))
            table.add_hline()
            table.end_table_last_footer()

            for i in range(data.shape[0]):
                table.add_row(data[i])

        # with self.doc.create(Table(position="htbp")) as table:
        #     table.append(NoEscape(r'\centering'))
        #
        #     tabular = Tabular(tab_style)
        #     tabular.add_hline()
        #     tabular.add_row(["header 1", "header 2", "header 3"])
        #     tabular.add_hline()
        #
        #     for i in range(150):
        #         tabular.add_row(row)
        #     tabular.add_hline()
        #
        #     table.append(tabular)
        #     table.add_caption('Caption Table {}'.format('Caption'))
        #     table.append(Label('table:{}'.format('tab1')))

    def write_test_page(self, sname):
        self.doc.append(NewPage())
        with self.doc.create(Section('Test')):
            # Reference
            with self.doc.create(Subsection('Reference')):
                self.doc.append(Command('cite', options=[], arguments='bertram'))
                self.doc.append(LineBreak())
                self.doc.append(Command('cite', options=[], arguments='simon06'))
                self.doc.append(LineBreak())

            # List
            with self.doc.create(Subsection('List')) as obj_sec:
                with self.doc.create(Itemize()) as itemize:
                    itemize.add_item("the first item")
                    itemize.add_item("the second item")
                    itemize.add_item("the third etc")
                    # you can append to existing items
                    itemize.append(Command("ldots"))
                    obj_sec.append(itemize)

            # Figure, matplotlib
            with self.doc.create(Subsection('Figure plot')) as obj_sec:
                name = 'fig1'
                caption = 'Caption Figure plot'
                self.doc.append(NoEscape('Fig. ' + Ref('figure:{}'.format(name)).dumps_as_content()))

                dpi = 3000

                x = [0, 1, 2, 3, 4, 5, 6]
                y = [15, 2, 7, 1, 5, 6, 9]
                plt.plot(x, y)

                self.insert_plot(obj_sec, name, caption, r'1\textwidth', dpi=dpi)

            # Figure, image
            with self.doc.create(Subsection('Figure image')) as obj_sec:
                name = 'fig2'
                caption = 'Caption Figure image'
                self.doc.append(NoEscape('Fig. ' + Ref('figure:{}'.format(name)).dumps_as_content()))

                self.insert_image(obj_sec, name, caption, r'1\textwidth', dpi=dpi)

            # Table, multi-page LongTable
            with self.doc.create(Subsection('Talbe')) as obj_sec:
                name = 'tab1'
                caption = 'Caption Table'
                self.doc.append(NoEscape('Tab. ' + Ref('table:{}'.format(name)).dumps_as_content()))

                header = ['header 1', 'header 2', 'header 3']
                # data = np.array([['1', 2, 3]])
                data = np.zeros((50, 3))

                self.insert_table(obj_sec, name, caption, r'1\textwidth', header, data)

            # Math, equation, numpy
            with self.doc.create(Subsection('Equation numpy')) as obj_sec:
                name = 'equ1'
                caption = 'Caption Equation 1'
                self.doc.append(NoEscape('Equ. ' + Ref('equation:{}'.format(name)).dumps_as_content()))

                M = np.asmatrix(
                    np.array([
                        [2, 3, 4],
                        [0, 0, 1],
                        [0, 0, 2]
                    ]))
                a = np.asmatrix(
                    np.array(
                        [
                            [100, 10, 20]
                        ]).T)

                with obj_sec.create(Alignat(aligns=2,
                                            numbering=True,
                                            escape=False)) as obj_agn:
                    # vec = Matrix(a)
                    # vec_name = VectorName(name)
                    # math = Math(data=[vec_name, '=', vec])
                    # obj_sec.append(math)
                    obj_agn.append(r'\frac{a}{b} &= 0 \\')
                    obj_agn.extend([Matrix(M), Matrix(a), '&=', Matrix(M * a)])
                    obj_agn.append(Label('equation:{}'.format(name)))
                    # print(obj_agn)

            # Math, equation, quantities
            with self.doc.create(Subsection('Equation quantities')) as obj_sec:
                name = 'equ2'
                self.doc.append(NoEscape('Equ. ' + Ref('equation:{}'.format(name)).dumps_as_content()))

                with obj_sec.create(Alignat(aligns=2,
                                            numbering=True,
                                            escape=False)) as obj_agn:
                    G = pq.constants.Newtonian_constant_of_gravitation
                    moon_earth_distance = 384400 * pq.km
                    moon_mass = 7.34767309e22 * pq.kg
                    earth_mass = 5.972e24 * pq.kg
                    moon_earth_force = G * moon_mass * earth_mass / moon_earth_distance ** 2
                    q1 = Quantity(moon_earth_force.rescale(pq.newton),
                                  options={
                                      'round-precision': 4,
                                      'round-mode': 'figures'})
                    # math = Math(data=['F=', q1])
                    # obj_sec.append(math)

                    obj_agn.extend(['F', '&=', q1])
                    obj_agn.append(Label('equation:{}'.format(name)))
                    # print(obj_agn)

        self.doc.append(Command('cleardoublepage'))

    def write_cover_page(self, sname):
        # fmt_date = "%d %b %Y"

        key = 'cover'
        print('{}'.format(key))

        opt = self.data['content'][key]
        if key in self.__conf['data']['content'].keys():
            if self.__conf['data']['content'][key] is not None:
                opt = self.__conf['data']['content'][key]

        doc_page = PageStyle(sname)

        doc_page.append(opt['title'])

        doc_page.append(Command('thispagestyle', 'empty'))
        self.doc.preamble.append(doc_page)
        self.doc.change_document_style(sname)
        self.doc.append(Command('cleardoublepage'))

    def write_title_page(self, sname):
        fmt_date = "%d %b %Y"
        self.doc.append(NewPage())

        key = 'title'
        print('{}'.format(key))

        opt = self.data['content'][key]
        if key in self.__conf['data']['content'].keys():
            if self.__conf['data']['content'][key] is not None:
                opt = self.__conf['data']['content'][key]

        doc_page = PageStyle(sname)

        doc_page.append(Command('title', NoEscape(opt['title'])))
        doc_page.append(Command('author', opt['author']))
        if isinstance(opt['date'], str):
            doc_page.append(
                Command('date',
                        datetime.strptime(opt['date'],
                                          '%Y-%m-%d').strftime(fmt_date)))
        elif isinstance(opt['date'], date):
            doc_page.append(
                Command('date',
                        opt['date'].strftime(fmt_date)))
        elif isinstance(opt['date'], datetime):
            doc_page.append(
                Command('date',
                        opt['date'].strftime(fmt_date)))
        else:
            doc_page.append(
                Command('date',
                        NoEscape(r'\today')))
        doc_page.append(Command('maketitle'))

        doc_page.append(Command('thispagestyle', 'empty'))
        self.doc.preamble.append(doc_page)
        self.doc.change_document_style(sname)
        self.doc.append(Command('cleardoublepage'))

    def write_toc_page(self, sname):
        self.doc.append(NewPage())

        doc_page = PageStyle(sname)

        # doc_page.append(NoEscape(r'\pdfbookmark[0]{\contentsname}{toc}'))
        doc_page.append(Command('tableofcontents'))

        doc_page.append(Command('thispagestyle', 'empty'))
        self.doc.preamble.append(doc_page)
        self.doc.change_document_style(sname)
        self.doc.append(Command('cleardoublepage'))

    def write_lof_page(self, sname):
        self.doc.append(NewPage())

        # self.doc.append(NoEscape(r'\pdfbookmark[0]{Figures}{lof}'))
        self.doc.append(Command('listoffigures'))

        self.doc.append(Command('addcontentsline',
                                ['toc', 'section', NoEscape(r'\listfigurename')]))
        self.doc.append(Command('cleardoublepage'))

    def write_lot_page(self, sname):
        self.doc.append(NewPage())

        # doc_page.append(NoEscape(r'\pdfbookmark[0]{Tables}{lot}'))
        self.doc.append(Command('listoftables'))

        self.doc.append(Command('addcontentsline',
                                ['toc', 'section', NoEscape(r'\listtablename')]))
        self.doc.append(Command('cleardoublepage'))

    def write_acknowledgment_page(self, sname):
        key = 'acknowledgement'
        print('{}'.format(key))

        self.doc.append(NewPage())
        self.doc.append(Command('RaggedRight'))

        opt = self.data['content'][key]
        with self.doc.create(Section(opt['title'], numbering=False)):
            for i in opt['paragraph'].keys():
                self.doc.append(opt['paragraph'][i])
                self.doc.append(LineBreak())

        self.doc.append(Command('addcontentsline',
                                ['toc', 'section', opt['title']]))
        self.doc.append(Command('cleardoublepage'))

    def write_abbreviation_page(self, sname):
        key = 'abbreviation'
        print('{}'.format(key))

        self.doc.append(NewPage())
        self.doc.append(Command('RaggedRight'))

        opt = self.data['content'][key]
        with self.doc.create(Section(opt['title'], numbering=False)) as obj_sec:
            self.insert_abbre(obj_sec, r'1\textwidth', opt['paragraph'])

        self.doc.append(Command('addcontentsline',
                                ['toc', 'section', opt['title']]))
        self.doc.append(Command('cleardoublepage'))

    def write_summary_page(self, sname):
        key = 'summary'
        print('{}'.format(key))

        self.doc.append(NewPage())
        self.doc.append(Command('RaggedRight'))

        opt = self.data['content'][key]
        with self.doc.create(Section(opt['title'], numbering=False)):
            for i in opt['paragraph'].keys():
                self.doc.append(opt['paragraph'][i])
                self.doc.append(LineBreak())

        self.doc.append(Command('addcontentsline',
                                ['toc', 'section', opt['title']]))
        self.doc.append(Command('cleardoublepage'))

    def write_section_page(self, sname):
        opt = self.data['content']['section']
        try:
            opt_sect = self.__conf['data']['content']['section']
            opt_data = self.__conf['data']['content']['data']
        except KeyError:
            raise KeyError

        # print(opt_content)
        for key_l1 in opt.keys():
            self.doc.append(NewPage())
            self.doc.append(Command('RaggedRight'))

            val_l1 = opt[key_l1]
            # val_l1_keys = val_l1.keys()
            val_l1_t = val_l1['title']
            val_l1_p = val_l1['paragraph']
            print('{}'.format(val_l1_t))

            with self.doc.create(Section(val_l1_t)):
                for key, val in val_l1_p.items():
                    print(' {}'.format(key))

                    # self.doc.append(val.format_map(opt_data))
                    self.doc.append(val.format(data=opt_data))
                    self.doc.append(LineBreak())

                # key_l2 = val_l1.keys()
                # key_l2.remove('title')
                # key_l2.remove('paragraph')
                for key_l2 in val_l1.keys():
                    if key_l2 != 'title' and key_l2 != 'paragraph':

                        val_l2 = val_l1[key_l2]
                        val_l2_t = val_l2['title']
                        val_l2_p = val_l2['paragraph']
                        print(' {}'.format(val_l2_t))

                        with self.doc.create(Subsection(val_l2_t)):
                            for key, val in val_l2_p.items():
                                print('  {}'.format(key))

                                # self.doc.append(val.format_map(opt_data))
                                self.doc.append(val.format(data=opt_data))
                                self.doc.append(LineBreak())

    def write_reference_page(self, sname):
        self.doc.append(NewPage())

        key = 'reference'
        files = []
        print('{}'.format(key))

        opt = self.data['content'][key]
        # file = os.path.join(self.path, opt['file'])
        file = NoEscape(os.path.join(self.path, opt['file']))
        # file = os.path.join(self.path, opt['file']).replace(os.sep, '/')
        # file = NoEscape(os.path.join(self.path, opt['file']).replace(os.sep, '/'))

        # files.append(os.path.splitext(file)[0])
        # files.append(NoEscape(os.path.splitext(file)[0]))
        if key in self.__conf['data']['content'].keys():
            if 'file' in self.__conf['data']['content'][key].keys():
                opt['file'] = self.__conf['data']['content'][key]['file']
                # file = opt['file']
                file = NoEscape(opt['file'])
                # file = './{}'.format(opt['file'])
                # file = NoEscape('./{}'.format(opt['file']))

                # files.append(os.path.splitext(file)[0])
                # files.append(NoEscape(os.path.splitext(file)[0]))
        # print(files)

        self.doc.preamble.append(Command('addbibresource',
                                         options=[],
                                         arguments=[file]))
        # self.doc.preamble.append(Command('bibliography', arguments=files))

        with self.doc.create(Section(opt['title'], numbering=False)):
            self.doc.append(Command('printbibliography',
                                    options=[
                                        'heading=none'],
                                    arguments=[]))

        self.doc.append(Command('addcontentsline',
                                ['toc', 'section', opt['title']]))
        self.doc.append(Command('cleardoublepage'))

    def write_annex_page(self, sname):
        self.doc.append(NewPage())

        key = 'annex'
        print('{}'.format(key))

        opt = self.data['content'][key]
        if key in self.__conf['data']['content'].keys():
            if 'file' in self.__conf['data']['content'][key].keys():
                opt['file'] = self.__conf['data']['content'][key]['file']
                file = NoEscape(opt['file'])

        with self.doc.create(Section(opt['title'], numbering=False)):
            self.doc.append(file)
            self.doc.append(LineBreak())

        self.doc.append(Command('addcontentsline',
                                ['toc', 'section', opt['title']]))
        self.doc.append(Command('cleardoublepage'))

    def saveas(self):
        fname = self.__conf['data']['doc']['name']
        ftypes = self.__conf['data']['doc']['saveas']

        file = os.path.join(self.__conf['path'], '{n}'.format(n=fname))
        file_with_ext = '{}.{}'.format(file, 'tex'.lower())
        print('Saving: "{}"'.format(file_with_ext))
        if os.path.isfile(file_with_ext):
            try:
                os.remove(file_with_ext)
            except PermissionError as err:
                print('Could not delete "{}"'.format(file_with_ext))
                os._exit(1)
        self.doc.generate_tex(file)

        # if file != '.bib' os.path.splitext(file)[0]

        for ftype in ftypes:
            file_with_ext = '{}.{}'.format(file, str(ftype).lower())
            print('Saving: "{}"'.format(file_with_ext))
            if str(ftype).upper() == 'PDF':
                if os.path.isfile(file_with_ext):
                    try:
                        os.remove(file_with_ext)
                    except PermissionError as err:
                        print('Could not delete "{}"'.format(file_with_ext))
                        os._exit(1)
                # pkg natbib will cause latex style error!
                # Opt 1
                # self.doc.generate_pdf(file, clean=True, clean_tex=False)
                self.doc.generate_pdf(file, clean=False, clean_tex=False)

                # Opt 2
                # .py  -> latexmk  -> .tex & .run.xml
                # .tex -> pdflatex -> .bcf
                # .bcf -> biber    -> .bbl
                # .bbl -> pdflatex -> .pdf 1st time
                # .bbl -> pdflatex -> .pdf 2ed time, make sure
                #
                # import subprocess
                # commands = [
                #     ['pdflatex', file + '.tex'],
                #     ['biber', file],
                #     # ['bibtex', file + '.aux'],
                #     ['pdflatex', file + '.tex'],
                #     ['pdflatex', file + '.tex']
                # ]
                # for c in commands:
                #     subprocess.call(c)

    def close(self):
        # if self.doc is not None:
        #     self.doc = None
        pass
