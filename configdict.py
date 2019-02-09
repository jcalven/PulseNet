from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def _maybe_convert_dict(value):
    if isinstance(value, dict):
        return ConfigDict(value)
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