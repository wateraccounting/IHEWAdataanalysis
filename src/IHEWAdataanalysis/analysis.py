# -*- coding: utf-8 -*-
"""
**Download**

Before use this module, create ``accounts.yml`` file.
And edit account information in the file.
"""
import os
import importlib
import datetime

import yaml

# PyCharm
# if __name__ == "__main__":
#
# >>> from .base import Base
# ImportError: cannot import name 'base' from '__main__'
# But works for setup.py
#
# >>> from base import Base
# ModuleNotFoundError
#
# PyCharm->Project Structure->"Sources": WaterAccounting\""
# from src.IHEWAdataanalysis.base.base import Base
# OK
#
# PyCharm->Project Structure->"Sources": IHEWAdataanalysis\"src\IHEWAdataanalysis"
# >>> from base import Base
# OK

try:
    # IHEClassInitError, IHEStringError, IHETypeError, IHEKeyError, IHEFileError
    from .exception import IHEClassInitError
except ImportError:
    from IHEWAdataanalysis.exception import IHEClassInitError


class Base(object):
    """This Base class

    Load base.yml file.

    Args:
        product (str): Product name of data products.
        is_print (bool): Is to print status message.
    """
    def __init__(self):
        """Class instantiation
        """
        pass

    def _scan_templates(self):
        pass


class Analysis(Base):
    """Download class

    After initialise the class, data downloading will automatically start.

    Args:
        workspace (str): Directory to config.yml.
        config (str): Configuration yaml file name.
        kwargs (dict): Other arguments.
    """
    def __init__(self, workspace='', config='', **kwargs):
        """Class instantiation
        """
        self.must_keys = {
            'template': ['provider', 'name'],
            'areas': []
        }

        self.__status = {
            'code': 0
        }
        self.__conf = {
            'path': '',
            'name': '',
            'time': {
                'start': None,
                'now': None,
                'end': None
            },
            'data': {
                # 'template': None,
                # 'data': None
            }
        }
        self.__tmp = {
            'name': '',
            'module': None
        }

        if isinstance(workspace, str):
            path = os.path.join(workspace)
            if not os.path.exists(path):
                os.makedirs(path)
            self.__conf['path'] = path
        else:
            self.__status['code'] = 1

        if isinstance(config, str):
            self.__conf['name'] = config
        else:
            self.__status['code'] = 1

        if self.__status['code'] == 0:
            self.__status['code'] = self._time()
            self.__status['code'] = self._conf()

        if self.__status['code'] == 0:
            self.__status['code'] = self._template()

        if self.__status['code'] == 0:
            if self.__tmp['module'] is None:
                self.__status['code'] = 1
            else:
                template = self.__tmp['module'].Template(self.__conf)
                # template.create()
                # template.write()
                # template.saveas()
                # template.close()

        if self.__status['code'] != 0:
            print('Status', self.__status['code'])
            raise IHEClassInitError('Analysis') from None

    def _conf(self) -> int:
        status_code = 0
        data = None

        file_conf = os.path.join(self.__conf['path'], self.__conf['name'])
        with open(file_conf) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)

        if data is not None:
            if data is not None:
                for key in self.must_keys.keys():
                    status_code += self._conf_keys(key, data)
                    # status_code += self._conf_keys('template', data)
        else:
            status_code = 1

        if status_code == 0:
            self.__conf['data'] = data

        return status_code

    def _conf_keys(self, key, data) -> int:
        status_code = 0
        try:
            if isinstance(data[key], dict):
               data_keys = data[key].keys()
            else:
                raise KeyError
        except KeyError:
            status_code = 1
        else:
            for data_key in self.must_keys[key]:
                if data_key not in data_keys:
                    status_code += 1

        return status_code

    def _time(self) -> int:
        """

        Returns:
            int: Status.
        """
        status_code = -1
        now = datetime.datetime.now()

        self.__conf['time']['start'] = now
        self.__conf['time']['now'] = now
        self.__conf['time']['end'] = now
        status_code = 0

        return status_code

    def _template(self) -> int:
        """

        Returns:
            dict: template.
        """
        status_code = -1
        template = self.__tmp
        module_name = template['name']
        module_obj = template['module']

        if self.__conf['data']['template'] is None:
            print('Please select a template!')

            status_code = 0

        else:
            try:
                module_provider = self.__conf['data']['template']['provider']
                module_template = self.__conf['data']['template']['name']
            except KeyError:
                status_code = 1
            else:
                module_name_base = '{tmp}.{nam}'.format(
                    tmp=module_provider,
                    nam=module_template)

                # Load module
                # module_obj = None
                if module_obj is None:
                    is_reload_module = False
                else:
                    if module_name == module_name_base:
                        is_reload_module = True
                    else:
                        is_reload_module = False
                template['name'] = module_name_base
                # print(template)

                if is_reload_module:
                    try:
                        module_obj = importlib.reload(module_obj)
                    except ImportError:
                        status_code = 1
                    else:
                        template['module'] = module_obj
                        print('Reloaded module.{nam}'.format(
                            nam=template['name']))
                        status_code = 0
                else:
                    try:
                        # importlib.import_module('.FAO', '.templates.IHE')
                        #
                        # importlib.import_module('templates.IHE.FAO')
                        #
                        # importlib.import_module('IHEWAdataanalysis.templates.IHE.FAO')
                        module_obj = \
                            importlib.import_module(
                                '.{n}'.format(n=module_template),
                                '.templates.{p}'.format(p=module_provider))
                        print('Loaded module from '
                              '.templates.{nam}'.format(nam=template['name']))
                    except ImportError:
                        module_obj = \
                            importlib.import_module(
                                'IHEWAdataanalysis.templates.{nam}'.format(
                                    nam=template['name']))
                        print('Loaded module from '
                              'IHEWAdataanalysis.templates.{nam}'.format(
                                  nam=template['name']))
                        status_code = 1
                    finally:
                        if module_obj is not None:
                            template['module'] = module_obj
                            status_code = 0
                        else:
                            status_code = 1

        # print(template)
        self.__tmp['name'] = template['name']
        self.__tmp['module'] = template['module']
        return status_code

    @staticmethod
    def get_config(self):
        print(self.__conf)
