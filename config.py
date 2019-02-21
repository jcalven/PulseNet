from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _maybe_convert_dict(value):
    if isinstance(value, dict):
        return ConfigDict(**value)
    return value


class ConfigDict(dict):
    """Configuration container class."""

    def __init__(self, **initial_dictionary):
        """Creates an instance of ConfigDict.
        Args:
          initial_dictionary: Optional dictionary or ConfigDict containing initial
            parameters.
        """
        if initial_dictionary:
            for field, value in initial_dictionary.items():
                initial_dictionary[field] = _maybe_convert_dict(value)
        super(ConfigDict, self).__init__(initial_dictionary)

    def __setattr__(self, attribute, value):
        self[attribute] = _maybe_convert_dict(value)

    def __getattr__(self, attribute):
        try:
            return self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __delattr__(self, attribute):
        try:
            del self[attribute]
        except KeyError as e:
            raise AttributeError(e)

    def __setitem__(self, key, value):
        super(ConfigDict, self).__setitem__(key, _maybe_convert_dict(value))
        
        
import configparser
import json

def config_reader(config_file, KEYS, entry=None):
    """
    Reads config (.ini) file, parses its contents and returns a dictionary with configuration 
    entries and values.
    Args:
        config_file (str): Path to configuration file.
        KEY (str): Section in configuration file to read.
        entry (str): Selects a single entry to return
    Returns:
        dict: Dictionary with configuration entries and values.
    """
    if not isinstance(KEYS, (list, tuple)):
        KEYS = [KEYS]
    conf_collection = {}
    for KEY in KEYS:
        #KEY = KEY.upper()
        config = configparser.ConfigParser(inline_comment_prefixes='#', empty_lines_in_values=False)
        config.read(config_file)
        conf = dict(config[KEY.upper()].items())
        for entry_val in conf:
            try:
                val = json.loads(config.get(KEY.upper(), entry_val))
            except:
                val = None
            conf.update({entry_val: val})
        if entry is None:
            conf = ConfigDict(**conf)
        else:
            conf = ConfigDict(**conf).get(entry.lower(), None)
        conf_collection[KEY.upper()] = conf
    if len(conf_collection) > 1:
        return conf_collection
    else:
        return conf_collection #.get(KEY, None)