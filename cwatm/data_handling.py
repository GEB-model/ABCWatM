# -------------------------------------------------------------------------
# Name:        Data handling
# Purpose:     Transforming netcdf to numpy arrays, checking mask file
#
# Author:      PB
#
# Created:     13/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.globals import (
    binding,
    option,
)


def returnBool(inBinding):
    """
    Test if parameter is a boolean and return an error message if not, and the boolean if everything is ok

    :param inBinding: parameter in settings file
    :return: boolean of inBinding
    """

    b = cbinding(inBinding)
    assert isinstance(b, bool)
    return b


def checkOption(inBinding):
    """
    Check if option in settings file has a counterpart in the source code

    :param inBinding: parameter in settings file
    """
    return option[inBinding]


def cbinding(inBinding):
    """
    Check if variable in settings file has a counterpart in the source code

    :param inBinding: parameter in settings file
    """
    return binding[inBinding]
