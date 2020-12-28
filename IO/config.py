"""
Author: Rex Geng
This file parses the given yaml file for usr specified argument
"""
import argparse
import os
from collections import OrderedDict

import ruamel.yaml


class EmptyConfig:
    def __init__(self):
        pass

    def __repr__(self):
        return '{}'


class UsrConfigs:
    def __init__(self, obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                setattr(self, k, UsrConfigs(v))
            elif isinstance(v, list) and isinstance(v[0], OrderedDict):
                setattr(self, k, [])
                for m in v:
                    getattr(self, k).append(UsrConfigs(m))
            elif v is None:
                setattr(self, k, EmptyConfig())
            else:
                setattr(self, k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for
                                      (k, v) in self.__dict__.items()))


def get_usr_config():
    parser = argparse.ArgumentParser(description='atr src API')
    parser.add_argument('usr_config', type=str)
    args = parser.parse_args()
    yaml_path = args.usr_config
    ext = os.path.splitext(yaml_path)[1].lstrip('.')
    ext = ext.rstrip(' ')
    assert ext == 'yaml'
    with open(yaml_path) as file:
        yml = ruamel.yaml.YAML()
        yml.allow_duplicate_keys = True
        doc = yml.load(file)
    usr_config = UsrConfigs(doc)
    return usr_config
