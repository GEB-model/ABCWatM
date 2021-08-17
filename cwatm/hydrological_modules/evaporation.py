# -------------------------------------------------------------------------
# Name:        Evaporation module
# Purpose:
#
# Author:      PB
#
# Created:     01/08/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules.data_handling import cbinding, checkOption
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass


class evaporation(object):
    """
    Evaporation module
    Calculate potential evaporation and pot. transpiration


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    cropKC                crop coefficient for each of the 4 different land cover types (forest, irrigated  --       
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        """The constructor evaporation"""
        self.var = model.data.landunit
        self.model = model
        
    def dynamic(self, ETRef):
        """
        Dynamic part of the soil module

        calculating potential Evaporation for each land cover class with kc factor
        get crop coefficient, use potential ET, calculate potential bare soil evaporation and transpiration

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        :return: potential evaporation from bare soil, potential transpiration
        """

        # get crop coefficient
        # to get ETc from ET0 x kc factor  ((see http://www.fao.org/docrep/X0490E/x0490e04.htm#TopOfPage figure 4:)
        # crop coefficient read for forest and grassland from file

        # calculate potential bare soil evaporation
        potBareSoilEvap = self.var.cropCorrect * self.var.minCropKC * ETRef
        # calculate snow evaporation
        self.var.snowEvap =  np.minimum(self.var.SnowMelt, potBareSoilEvap)
        self.var.SnowMelt = self.var.SnowMelt - self.var.snowEvap
        potBareSoilEvap = potBareSoilEvap - self.var.snowEvap

        # calculate potential ET
        ##  self.var.totalPotET total potential evapotranspiration for a reference crop for a land cover class [m]
        totalPotET = self.var.cropCorrect * self.var.cropKC * ETRef

        ## potTranspiration: Transpiration for each land cover class
        potTranspiration = np.maximum(0., totalPotET - potBareSoilEvap - self.var.snowEvap)

        return potTranspiration, potBareSoilEvap, totalPotET