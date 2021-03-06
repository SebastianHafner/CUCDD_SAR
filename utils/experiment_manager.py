# Largely taken from FVCore and Detectron2

import logging
from argparse import ArgumentParser
from tabulate import tabulate
from collections import OrderedDict
import yaml
from fvcore.common.config import CfgNode as _CfgNode
from pathlib import Path

# TODO Initialize Cfg from Base Config
class CfgNode(_CfgNode):
    """
    The same as `fvcore.common.config.CfgNode`, but different in:

    1. Use unsafe yaml loading by default.
      Note that this may lead to arbitrary code execution: you must not
      load a config file from untrusted sources before manually inspecting
      the content of the file.
    2. Support config versioning.
      When attempting to merge an old config, it will convert the old config automatically.

    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):

        # Always allow merging new configs
        self.__dict__[CfgNode.NEW_ALLOWED] = True
        super(CfgNode, self).__init__(init_dict, key_list, True)


    # Note that the default value of allow_unsafe is changed to True
    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:
        loaded_cfg = _CfgNode.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        # defaults.py needs to import CfgNode
        self.merge_from_other_cfg(loaded_cfg)

def new_config():
    '''
    Creates a new config based on the default config file
    :return:
    '''
    from .defaults import C
    return C.clone()

global_config = CfgNode()

class HPConfig():
    '''
    A hyperparameter config object
    '''
    def __init__(self):
        self.data = {}
        self.argparser = ArgumentParser()

    def create_hp(self, name, value, argparse=False, argparse_args={}):
        '''
        Creates a new hyperparameter, optionally sourced from argparse external arguments
        :param name:
        :param value:
        :param argparse:
        :param argparse_args:
        :return:
        '''
        self.data[name] = value
        if argparse:
            datatype = type(value)
            # Handle boolean type
            if datatype == bool:
                self.argparser.add_argument(f'--{name}', action='store_true', *argparse_args)
            else:
                self.argparser.add_argument(f'--{name}', type=datatype, *argparse_args)

    def parse_args(self):
        '''
        Performs a parse operation from the program arguments
        :return:
        '''
        args = self.argparser.parse_known_args()[0]
        for key, value in args.__dict__.items():
            # Arg not present, using default
            if value is None: continue
            self.data[key] = value

    def __str__(self):
        '''
        Converts the HP into a human readable string format
        :return:
        '''
        table = {'hyperparameter': self.data.keys(),
                'values': list(self.data.values()),
                 }
        return tabulate(table, headers='keys', tablefmt="fancy_grid", )


    def save_yml(self, file_path):
        '''
        Save HP config to a yaml file
        :param file_path:
        :return:
        '''
        with open(file_path, 'w') as file:
            yaml.dump(self.data, file, default_flow_style=False)

    def load_yml(self, file_path):
        '''
        Load HP Config from a yaml file
        :param file_path:
        :return:
        '''
        with open(file_path, 'r') as file:
            yml_hp = yaml.safe_load(file)

        for hp_name, hp_value in yml_hp.items():
            self.data[hp_name] = hp_value

    def __getattr__(self, name):
        return self.data[name]


def config(name='default') -> HPConfig:
    '''
    Retrives a configuration (optionally, creating it) of the run. If no `name` provided, then 'default' is used
    :param name: Optional name of the
    :return: HPConfig object
    '''
    # Configuration doesn't exist yet
    # if name not in _config_data.keys():
    #     _config_data[name] = HPConfig()
    # return _config_data[name]
    pass


# loading cfg
def load_cfg(yaml_file: Path):
    cfg = new_config()
    cfg.merge_from_file(str(yaml_file))
    cfg.NAME = str(yaml_file.stem)
    return cfg


def setup(args):
    cfg = new_config()
    cfg.merge_from_file(f'configs/{args.config_file}.yaml')
    cfg.merge_from_list(args.opts)
    cfg.NAME = args.config_file
    cfg.RESUME_CHECKPOINT = args.resume_checkpoint

    # TODO: might not be necessary -> remove
    if args.data_dir:
        cfg.DATASETS.TRAIN = (args.data_dir,)
    return cfg

import argparse

def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c',"--config-file", dest='config_file', default="", required=True, metavar="FILE", help="path to config file")
    parser.add_argument('-r', "--resume-checkpoint", dest='resume_checkpoint', type=int, default=0, help='checkpoint to resume training from')
    parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                      default='', help='dataset directory')
    parser.add_argument('-o', '--output-dir', dest='log_dir', type=str,
                      default='', help='output directory')
    parser.add_argument(
        "--resume",
        dest='resume',
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument('--resume-from', dest='resume_from', type=str,
                      default='', help='path of which the model will be loaded from')
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument('-sar_c', "--sar_config-file", dest='sar_config_file', default='sar', metavar="FILE",
                        help="path to sar config file")

    # Hacky hack
    # parser.add_argument("--eval-training", action="store_true", help="perform evaluation on training set only")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser
