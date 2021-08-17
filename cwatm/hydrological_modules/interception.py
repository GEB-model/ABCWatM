# -------------------------------------------------------------------------
# Name:        Interception module
# Purpose:
#
# Author:      PB
#
# Created:     01/08/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import loadmap, readnetcdf2, divideValues, checkOption
import numpy as np

class interception(object):
    """
    INTERCEPTION


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m        
    waterbalance_module                                                                                              
    interceptCap          interception capacity of vegetation                                               m        
    minInterceptCap       Maximum interception read from file for forest and grassland land cover           m        
    interceptStor         simulated vegetation interception storage                                         m        
    Rain                  Precipitation less snow                                                           m        
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m        
    SnowMelt              total snow melt from all layers                                                   m        
    interceptEvap         simulated evaporation from water intercepted by vegetation                        m        
    potTranspiration      Potential transpiration (after removing of evaporation)                           m        
    actualET              simulated evapotranspiration from soil, flooded area and vegetation               m        
    snowEvap              total evaporation from snow for a snow layers                                     m        
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.landunit
        self.model = model

    def initial(self):
        self.var.minInterceptCap = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.interceptStor = self.var.full_compressed(np.nan, dtype=np.float32)

        for coverNum, coverType in enumerate(self.model.coverTypes):
            coverType_indices = np.where(self.var.land_use_type == coverNum)
            self.var.minInterceptCap[coverType_indices] = self.model.data.to_landunit(data=loadmap(coverType + "_minInterceptCap"), fn=None)
            if coverType in ('forest', 'grassland', 'irrPaddy', 'irrNonPaddy', 'sealed'):
                initial = self.model.data.to_landunit(data=self.model.data.var.load_initial(coverType + "_interceptStor"), fn=None)
                if not isinstance(initial, float):
                    initial = initial[coverType_indices]
                self.var.interceptStor[coverType_indices] = initial
            else:
                self.var.interceptStor[coverType_indices] = 0  # 0 for water
        
        
        assert not np.isnan(self.var.interceptStor).any()
        assert not np.isnan(self.var.minInterceptCap).any()

    def dynamic(self, potTranspiration):
        """
        Dynamic part of the interception module
        calculating interception for each land cover class

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        :return: interception evaporation, interception storage, reduced pot. transpiration

        """

        if checkOption('calcWaterBalance'):
            interceptStor_pre = self.var.interceptStor.copy()

        interceptCap = self.var.full_compressed(np.nan, dtype=np.float32)
        for coverNum, coverType in enumerate(self.model.coverTypes):
            coverType_indices = np.where(self.var.land_use_type == coverNum)
            if coverType in ('forest', 'grassland'):
                covertype_interceptCapNC = readnetcdf2(coverType + '_interceptCapNC', globals.dateVar['10day'], "10day")
                covertype_interceptCapNC = self.model.data.to_landunit(data=covertype_interceptCapNC, fn=None)  # checked
                interceptCap[coverType_indices] = covertype_interceptCapNC[coverType_indices]
            else:
                interceptCap[coverType_indices] = self.var.minInterceptCap[coverType_indices]
        
        assert not np.isnan(interceptCap).any()

        # Rain instead Pr, because snow is substracted later
        # assuming that all interception storage is used the other time step
        throughfall = np.maximum(0.0, self.var.Rain + self.var.interceptStor - interceptCap)

        # update interception storage after throughfall
        self.var.interceptStor = self.var.interceptStor + self.var.Rain - throughfall

        # availWaterInfiltration Available water for infiltration: throughfall + snow melt
        self.var.natural_available_water_infiltration = np.maximum(0.0, throughfall + self.var.SnowMelt)

        sealed_area = np.where(self.var.land_use_type == 4)
        water_area = np.where(self.var.land_use_type == 5)
        bio_area = np.where(self.var.land_use_type < 4)  # 'forest', 'grassland', 'irrPaddy', 'irrNonPaddy'

        self.var.interceptEvap = self.var.full_compressed(np.nan, dtype=np.float32)
        # interceptEvap evaporation from intercepted water (based on potTranspiration)
        self.var.interceptEvap[bio_area] = np.minimum(
            self.var.interceptStor[bio_area],
            potTranspiration[bio_area] * divideValues(self.var.interceptStor[bio_area], interceptCap[bio_area]) ** (2./3.)
        )

        self.var.interceptEvap[sealed_area] = np.maximum(
            np.minimum(self.var.interceptStor[sealed_area], self.var.EWRef[sealed_area]),
            self.var.full_compressed(0, dtype=np.float32)[sealed_area]
        )

        self.var.interceptEvap[water_area] = 0  # never interception for water

        # update interception storage and potTranspiration
        self.var.interceptStor = self.var.interceptStor - self.var.interceptEvap
        potTranspiration = np.maximum(0, potTranspiration - self.var.interceptEvap)

        # update actual evaporation (after interceptEvap)
        # interceptEvap is the first flux in ET, soil evapo and transpiration are added later
        self.var.actualET = self.var.interceptEvap + self.var.snowEvap

        if checkOption('calcWaterBalance'):
            self.model.waterbalance_module.waterBalanceCheck(
                how='cellwise',
                influxes=[self.var.Rain, self.var.SnowMelt],  # In
                outfluxes=[self.var.natural_available_water_infiltration, self.var.interceptEvap],  # Out
                prestorages=[interceptStor_pre],  # prev storage
                poststorages=[self.var.interceptStor],
                tollerance=1e-7
            )

        # if self.model.args.use_gpu:
            # self.var.interceptEvap = self.var.interceptEvap.get()

        return potTranspiration