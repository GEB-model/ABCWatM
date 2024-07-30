# -------------------------------------------------------------------------
# Name:        MiscInitial
# Purpose:
#
# Author:      pb
#
# Created:     13.07.2016
# Copyright:   (c) pb 2016

# -------------------------------------------------------------------------

from cwatm.management_modules.data_handling import *


class miscInitial(object):
    """
    Miscellaneous repeatedly used expressions
    Definition if cell area comes from regular grid e.g. 5x5km or from irregular lat/lon
    Conversion factors between m3 and mm etc.

    Note:
        Only used in the initial phase.


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    DtSec                 number of seconds per timestep (default = 86400)                                  s
    MtoM3                 Coefficient to change units                                                       --
    InvDtSec
    cellArea              Cell area [m²] of each simulated mesh
    cellLength            length of a grid cell                                                             m
    InvCellArea           Inverse of cell area of each simulated mesh                                       m-1
    DtDay                 seconds in a timestep (default=86400)                                             s
    InvDtDay              inverse seconds in a timestep (default=86400)                                     s-1
    MMtoM                 Coefficient to change units                                                       --
    MtoMM                 Coefficient to change units                                                       --
    M3toM                 Coefficient to change units                                                       --
    con_precipitation     conversion factor for precipitation                                               --
    con_e                 conversion factor for evaporation                                                 --
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.grid
        self.HRU = model.data.HRU
        self.model = model

    def initial(self):
        """
        Initialization of some basic parameters e.g. cellArea

        * grid area, length definition
        * conversion factors
        * conversion factors for precipitation and pot evaporation
        """

        #            self.var.PixelArea = spatial(self.var.PixelArea)
        # Convert to spatial expresion (otherwise this variable cannnot be
        # used in areatotal function)

        # -----------------------------------------------------------------
        # Miscellaneous repeatedly used expressions (as suggested by GF)

        # self.var.InvCellLength = 1.0 / self.var.cellLength
        # self.var.InvCellArea = 1.0 / self.var.cellArea
        # self.HRU.InvCellArea = 1.0 / self.HRU.cellArea  # checked
        # Inverse of pixel size [1/m]
        self.model.DtSec = 86400.0
        self.model.DtDay = self.model.DtSec / 86400
        # Time step, expressed as fraction of day (used to convert
        # rate variables that are expressed as a quantity per day to
        # into an amount per time step)
        self.model.InvDtSec = 1 / self.model.DtSec
        # Inverse of time step [1/d]
