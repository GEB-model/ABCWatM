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
except (ModuleNotFoundError, ImportError):
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
        self.var = model.data.HRU
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
        land_use_indices_forest = np.where(self.var.land_use_type == 0) 
        land_use_indices_grassland = np.where(self.var.land_use_type == 1) 
        land_use_indices_agriculture = np.where((self.var.land_use_type == 2) | (self.var.land_use_type == 3)) 
        if self.model.config["general"]["name"] == "100 infiltration change" or self.model.config["general"]["name"] == "restoration opportunities":
            if self.var.indices_agriculture_land_use_change[0].size != land_use_indices_agriculture[0].size:
                import rioxarray
                HRUs_to_forest = self.var.HRUs_to_forest
                self.var.land_use_type[HRUs_to_forest] = 0  # 0 is forest

        potBareSoilEvap = self.var.cropCorrect * self.var.minCropKC * ETRef 

        # calculate snow evaporation
        self.var.snowEvap =  np.minimum(self.var.SnowMelt, potBareSoilEvap) 
        self.var.SnowMelt = self.var.SnowMelt - self.var.snowEvap

        # calculate potential ET
        ##  self.var.totalPotET total potential evapotranspiration for a reference crop for a land cover class [m] 


        if np.isnan(self.var.cropKC[land_use_indices_agriculture]).any():
            # Replace NaN values with the replacement value
            self.var.cropKC[land_use_indices_agriculture] = np.nanmean(self.var.cropKC[land_use_indices_grassland])

        totalPotET = self.var.cropCorrect * self.var.cropKC * ETRef 
        totalPotET[land_use_indices_forest] *= 1.2
        potBareSoilEvap[land_use_indices_forest] *= 0.425
        #totalPotET *= 
        

       
        # change pot ET of grasslands
        self.potET_forest = totalPotET[land_use_indices_forest]
        self.potET_grassland = totalPotET[land_use_indices_grassland]
        self.potET_agriculture = totalPotET[land_use_indices_agriculture]

        ## potTranspiration: Transpiration for each land cover class
        potTranspiration = np.maximum(0., totalPotET - potBareSoilEvap - self.var.snowEvap)

        return potTranspiration, potBareSoilEvap, totalPotET
      # self.potET_forest,self.potET_grassland,self.potET_agriculture