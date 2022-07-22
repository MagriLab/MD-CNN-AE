'''Assorted functions not related to machine learning.'''

import configparser
import typing
import pathlib

StrOrPath = typing.Union[str, pathlib.Path]

class GetAttr(object):
    '''Convert a dictionary to data class

    The values of the dictionary can be accessed as self.key
    '''
    def __init__(self, _dict):
        self.__dict__.update(_dict)

def read_config_ini(f:StrOrPath) -> configparser.ConfigParser:
    '''Read a .ini config file and returns the ConfigParser object.'''
    config = configparser.ConfigParser()
    config.read(f)
    return config