"""Utilities used for dsim operation."""

import os
from configparser import SafeConfigParser
from typing import Tuple

from casperfpga.network import IpAddress


def get_prefixed_name(input_str: str, prefix: str) -> str:
    """
    Extract the name from a string with a known prefix attached to it.

    :param full_str:
        Full string with prefix.
    :param prefix:
        Prefix to extract from full_str.

    :return:
        Extracted string.
    """
    if not input_str.startswith(prefix):
        return None
    if input_str.startswith(prefix + "_"):
        return input_str[len(prefix) + 1 :]
    else:
        return input_str[len(prefix) :]


def remove_nones(write_vars: dict) -> dict:
    """Remove None-types from a dictionary."""
    return {k: v for k, v in list(write_vars.items()) if v is not None}


def parse_config_file(config_file: str = "") -> dict:
    """
    Parse a config.ini file into a dictionary.

    :param config_file:
        The ini file to process.

    :return:
        A dictionary containing the configuration.
    """
    if config_file == "" or config_file is None:
        raise ValueError("No config file given to be parsed...")
    else:
        # File was given, check if it exists
        if not (os.path.exists(config_file) and os.path.isfile(config_file)):
            # Problem
            errmsg = f"Config file given ({config_file}) is not valid!"
            raise ValueError(errmsg)

    parser = SafeConfigParser()
    files = parser.read(config_file)
    if len(files) == 0:
        errmsg = f"Could not read the config file: {config_file}"
        raise IOError(errmsg)

    config = {}
    for section in parser.sections():
        config[section] = {}
        for items in parser.items(section):
            config[section][items[0]] = items[1]

    return config


class StreamAddress(object):
    """
    A data source from an IP. Holds all the information we need to use that data source.

    :param ip_string:
        Address at which it can be found - a dotted decimal string.
    :param ip_range:
        The consecutive number of IPs over which it is spread.
    :param port:
        Port that data is sent to.
    """

    def __init__(self, ip_string: str, ip_range: int, port: int):
        self.ip_address = IpAddress(ip_string)
        self.ip_range = ip_range
        self.port = port

    @staticmethod
    def _parse_address_string(address_str: str) -> Tuple[str, int, int]:
        """
        Parse an IP address input as a string.

        :param address_string:
            IP address given in the form 1.2.3.4+50:6666.
        :return:
            Three-tuple with address, range and port numbers.
        """
        try:
            _bits = address_str.split(":")
            port = int(_bits[1])
            if "+" in _bits[0]:
                address, number = _bits[0].split("+")
                number = int(number) + 1
            else:
                address = _bits[0]
                number = 1
            assert len(address.split(".")) == 4
        except ValueError:
            raise RuntimeError(f"Address {address_str} is not correctly formed. Expecting e.g. 1.2.3.4+50:6666.")
        return address, number, port

    @classmethod
    def from_address_string(cls, address_string: str) -> "StreamAddress":
        """
        Parse an IP address input as a string and return a StreamAddress object.

        :param address_string:
            IP address given in the form 1.2.3.4+50:6666.
        :return:
            StreamAddress object.
        """
        address, number, port = StreamAddress._parse_address_string(address_string)

        return cls(address, number, port)

    def is_multicast(self) -> bool:
        """Check if this data source is a multicast source (begins with 239.xx)."""
        return self.ip_address.is_multicast()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        return "%s+%i:%i" % (self.ip_address, self.ip_range - 1, self.port)

    def __str__(self):
        return "%s+%i:%i" % (self.ip_address, self.ip_range - 1, self.port)
