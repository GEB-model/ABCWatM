# -------------------------------------------------------------------------
# Name:        Waterdemand modules
# Purpose:
#
# Author:      PB, YS, MS, JdB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass


class waterdemand_irrigation:
    """
    WATERDEMAND

    calculating water demand - irrigation
    Agricultural water demand based on water need by plants

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    cropKC                crop coefficient for each of the 4 different land cover types (forest, irrigated  --       
    load_initial                                                                                                     
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m        
    fracVegCover          Fraction of area covered by the corresponding landcover type                               
    ws1                   Maximum storage capacity in layer 1                                               m        
    ws2                   Maximum storage capacity in layer 2                                               m        
    wfc1                  Soil moisture at field capacity in layer 1                                                 
    wfc2                  Soil moisture at field capacity in layer 2                                                 
    wwp1                  Soil moisture at wilting point in layer 1                                                  
    wwp2                  Soil moisture at wilting point in layer 2                                                  
    w1                    Simulated water storage in the layer 1                                            m        
    w2                    Simulated water storage in the layer 2                                            m        
    topwater              quantity of water above the soil (flooding)                                       m        
    arnoBeta                                                                                                         
    maxtopwater           maximum heigth of topwater                                                        m        
    totAvlWater                                                                                                      
    InvCellArea           Inverse of cell area of each simulated mesh                                       m-1      
    totalPotET            Potential evaporation per land use class                                          m        
    unmetDemandPaddy                                                                                                 
    unmetDemandNonpaddy                                                                                              
    unmetDemand                                                                                                      
    efficiencyPaddy                                                                                                  
    efficiencyNonpaddy                                                                                               
    returnfractionIrr                                                                                                
    pot_irrConsumption                                                                                               
    irrDemand                                                                                                        
    totalIrrDemand                                                                                                   
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

    def initial(self):
        """
        Initial part of the water demand module
        irrigation

        """

    def dynamic(self, totalPotET):
        """
        Dynamic part of the water demand module

        * calculate the fraction of water from surface water vs. groundwater
        * get non-Irrigation water demand and its return flow fraction
        """
        pot_irrConsumption = self.var.full_compressed(0, dtype=np.float32)
        # Paddy irrigation -> No = 2
        # Non paddy irrigation -> No = 3

        # a function of cropKC (evaporation and transpiration) and available water see Wada et al. 2014 p. 19        
        paddy_irrigated_land = np.where(self.var.land_use_type == 2)
        pot_irrConsumption[paddy_irrigated_land] = np.where(
            self.var.cropKC[paddy_irrigated_land] > 0.75,
            np.maximum(
                0.,
                (self.var.maxtopwater - (self.var.topwater[paddy_irrigated_land] + self.var.natural_available_water_infiltration[paddy_irrigated_land]))
            ), 
            0.)

        nonpaddy_irrigated_land = np.where(self.var.land_use_type == 3)[0]

        # Infiltration capacity
        #  ========================================
        # first 2 soil layers to estimate distribution between runoff and infiltration
        soilWaterStorage = self.var.w1[nonpaddy_irrigated_land] + self.var.w2[nonpaddy_irrigated_land]
        soilWaterStorageCap = self.var.ws1[nonpaddy_irrigated_land] + self.var.ws2[nonpaddy_irrigated_land]
        relSat = soilWaterStorage / soilWaterStorageCap
        satAreaFrac = 1 - (1 - relSat) ** self.var.arnoBeta[nonpaddy_irrigated_land]
        satAreaFrac = np.maximum(np.minimum(satAreaFrac, 1.0), 0.0)

        store = soilWaterStorageCap / (self.var.arnoBeta[nonpaddy_irrigated_land] + 1)
        potBeta = (self.var.arnoBeta[nonpaddy_irrigated_land] + 1) / self.var.arnoBeta[nonpaddy_irrigated_land]
        potInfiltrationCapacity = store - store * (1 - (1 - satAreaFrac) ** potBeta)
        # ----------------------------------------------------------
        availWaterPlant1 = np.maximum(0., self.var.w1[nonpaddy_irrigated_land] - self.var.wwp1[nonpaddy_irrigated_land])   #* self.var.rootDepth[0][No]  should not be multiplied again with soildepth
        availWaterPlant2 = np.maximum(0., self.var.w2[nonpaddy_irrigated_land] - self.var.wwp2[nonpaddy_irrigated_land])   # * self.var.rootDepth[1][No]
        #availWaterPlant3 = np.maximum(0., self.var.w3[No] - self.var.wwp3[No])  #* self.var.rootDepth[2][No]
        readAvlWater = availWaterPlant1 + availWaterPlant2 # + availWaterPlant3

        # calculate   ****** SOIL WATER STRESS ************************************

        #The crop group number is a indicator of adaptation to dry climate,
        # e.g. olive groves are adapted to dry climate, therefore they can extract more water from drying out soil than e.g. rice.
        # The crop group number of olive groves is 4 and of rice fields is 1
        # for irrigation it is expected that the crop has a low adaptation to dry climate
        #cropGroupNumber = 1.0
        etpotMax = np.minimum(0.1 * (totalPotET[nonpaddy_irrigated_land] * 1000.), 1.0)
        # to avoid a strange behaviour of the p-formula's, ETRef is set to a maximum of 10 mm/day.

        # for group number 1 -> those are plants which needs irrigation
        # p = 1 / (0.76 + 1.5 * np.minimum(0.1 * (self.var.totalPotET[No] * 1000.), 1.0)) - 0.10 * ( 5 - cropGroupNumber)
        p = 1 / (0.76 + 1.5 * etpotMax) - 0.4
        # soil water depletion fraction (easily available soil water) # Van Diepen et al., 1988: WOFOST 6.0, p.87.
        p = p + (etpotMax - 0.6) / 4
        # correction for crop group 1  (Van Diepen et al, 1988) -> p between 0.14 - 0.77
        p = np.maximum(np.minimum(p, 1.0), 0.)
        # p is between 0 and 1 => if p =1 wcrit = wwp, if p= 0 wcrit = wfc
        # p is closer to 0 if evapo is bigger and cropgroup is smaller

        wCrit1 = ((1 - p) * (self.var.wfc1[nonpaddy_irrigated_land] - self.var.wwp1[nonpaddy_irrigated_land])) + self.var.wwp1[nonpaddy_irrigated_land]
        wCrit2 = ((1 - p) * (self.var.wfc2[nonpaddy_irrigated_land] - self.var.wwp2[nonpaddy_irrigated_land])) + self.var.wwp2[nonpaddy_irrigated_land]
        # wCrit3 = ((1 - p) * (self.var.wfc3[No] - self.var.wwp3[No])) + self.var.wwp3[No]

        critWaterPlant1 = np.maximum(0., wCrit1 - self.var.wwp1[nonpaddy_irrigated_land])  # * self.var.rootDepth[0][No]
        critWaterPlant2 = np.maximum(0., wCrit2 - self.var.wwp2[nonpaddy_irrigated_land])  # * self.var.rootDepth[1][No]
        #critWaterPlant3 = np.maximum(0., wCrit3 - self.var.wwp3[No]) # * self.var.rootDepth[2][No]
        critAvlWater = critWaterPlant1 + critWaterPlant2 # + critWaterPlant3

        pot_irrConsumption[nonpaddy_irrigated_land] = np.where(
            self.var.cropKC[nonpaddy_irrigated_land] > 0.20,
            np.where(
                readAvlWater < critAvlWater,
                np.maximum(0.0, self.var.totAvlWater[nonpaddy_irrigated_land] - readAvlWater),
                0.
            ),
            0.
        )
        # should not be bigger than infiltration capacity
        pot_irrConsumption[nonpaddy_irrigated_land] = np.minimum(pot_irrConsumption[nonpaddy_irrigated_land], potInfiltrationCapacity)
        
        return pot_irrConsumption