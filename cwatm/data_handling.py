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
)

def cbinding(inBinding):
    """
    Check if variable in settings file has a counterpart in the source code

    :param inBinding: parameter in settings file
    """
    return binding[inBinding]
