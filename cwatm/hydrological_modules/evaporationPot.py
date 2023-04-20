# -------------------------------------------------------------------------
# Name:        POTENTIAL REFERENCE EVAPO(TRANSPI)RATION
# Purpose:
#
# Author:      PB
#
# Created:     10/01/2017
# Copyright:   (c) PB 2017
# -------------------------------------------------------------------------

import numpy as np
from cwatm.management_modules.data_handling import loadmap, returnBool, readnetcdf2, loadmap
from cwatm.management_modules.globals import dateVar, binding


class evaporationPot(object):
    """
    POTENTIAL REFERENCE EVAPO(TRANSPI)RATION
    Calculate potential evapotranspiration from climate data mainly based on FAO 56 and LISVAP
    Based on Penman Monteith

    References:
        http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation
        http://www.fao.org/docrep/X0490E/x0490e06.htm  http://www.fao.org/docrep/X0490E/x0490e06.htm
        https://ec.europa.eu/jrc/en/publication/eur-scientific-and-technical-research-reports/lisvap-evaporation-pre-processor-lisflood-water-balance-and-flood-simulation-model

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    cropCorrect           calibrated factor of crop KC factor                                               --       
    pet_modus             Flag: index which ETP approach is used e.g. 1 for Penman-Monteith                 --       
    AlbedoCanopy          Albedo of vegetation canopy (FAO,1998) default =0.23                              --       
    AlbedoSoil            Albedo of bare soil surface (Supit et. al. 1994) default = 0.15                   --       
    AlbedoWater           Albedo of water surface (Supit et. al. 1994) default = 0.05                       --       
    co2                                                                                                              
    TMin                  minimum air temperature                                                           K        
    TMax                  maximum air temperature                                                           K        
    Psurf                 Instantaneous surface pressure                                                    Pa       
    Qair                  specific humidity                                                                 kg/kg    
    Tavg                  average air Temperature (input for the model)                                     K        
    Rsdl                  long wave downward surface radiation fluxes                                       W/m2     
    albedoLand            albedo from land surface (from GlobAlbedo database)                               --       
    albedoOpenWater       albedo from open water surface (from GlobAlbedo database)                         --       
    Rsds                  short wave downward surface radiation fluxes                                      W/m2     
    Wind                  wind speed                                                                        m/s      
    ETRef                 potential evapotranspiration rate from reference crop                             m        
    EWRef                 potential evaporation rate from water surface                                     m        
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        """
        The constructor evaporationPot
        """
        self.var = model.data.HRU
        self.model = model
        
    def initial(self):
        """
        Initial part of evaporation type module
        Load inictial parameters

        Note:
            Only run if *calc_evaporation* is True
        """

        #self.var.sumETRef = globals.inZero.copy()
        self.var.cropCorrect = loadmap('crop_correct')
        self.var.cropCorrect = self.model.data.to_HRU(data=self.var.cropCorrect, fn=None)
        
        self.var.AlbedoCanopy = loadmap('AlbedoCanopy')
        self.var.AlbedoSoil = loadmap('AlbedoSoil')
        self.var.AlbedoWater = loadmap('AlbedoWater')

    def dynamic(self):
        """
        Dynamic part of the potential evaporation module
        Based on Penman Monteith - FAO 56

        """
        ESatmin = 0.6108* np.exp((17.27 * self.var.TMin) / (self.var.TMin + 237.3))
        ESatmax = 0.6108* np.exp((17.27 * self.var.TMax) / (self.var.TMax + 237.3))
        ESat = (ESatmin + ESatmax) / 2.0   # [KPa]
        # http://www.fao.org/docrep/X0490E/x0490e07.htm   equation 11/12
        RNup = 4.903E-9 * (((self.var.TMin + 273.16) ** 4) + ((self.var.TMax + 273.16) ** 4)) / 2
        # Up longwave radiation [MJ/m2/day]
        LatHeatVap = 2.501 - 0.002361 * self.var.Tavg
        # latent heat of vaporization [MJ/kg]

        # --------------------------------
        # if only daily calculate radiation is given instead of longwave down and shortwave down radiation
        if 'only_radiation' in binding and returnBool('only_radiation'):
            # FAO 56 - https://www.fao.org/3/x0490E/x0490e07.htm#solar%20radiation  equation 39
            radian = np.pi / 180 * self.var.lat
            distanceSun = 1 + 0.033 * np.cos(2 * np.pi * dateVar['doy'] / 365)
            # Chapter 3: equation 24
            declin = 0.409 * np.sin(2 * np.pi * dateVar['doy'] / 365 - 1.39)
            ws = np.arccos(-np.tan(radian * np.tan(declin)))
            Ra = 24 *60 / np.pi * 0.082 * distanceSun * (ws * np.sin(radian) * np.sin(declin) + np.cos(radian) * np.cos(declin) * np.sin(ws))
            # Equation 21 Chapter 3
            Rso = Ra * (0.75 + (2 * 10 ** -5 * self.var.dem)) # in MJ/m2/day
            # Equation 37 Chapter 3
            RsRso = 1.35 * self.var.Rsds/Rso - 0.35
            RsRso = np.minimum(np.maximum(RsRso, 0.05), 1)
            EmNet = (0.34 - 0.14 * np.sqrt(self.var.EAct))  # Eact in hPa but needed in kPa : kpa = 0.1 * hPa - conversion done in readmeteo
            RLN = RNup * EmNet * RsRso
            # Equation 39 Chapter 3

            Psycon = 0.00163 * (101.3 / LatHeatVap)
            # psychrometric constant at sea level [mbar/deg C]
            #Psycon = 0.665E-3 * self.var.Psurf
            # psychrometric constant [kPa C-1]
            # http://www.fao.org/docrep/X0490E/x0490e07.htm  Equation 8
            # see http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation
            Psycon = Psycon * ((293 - 0.0065 * self.var.dem) / 293) ** 5.26  # in [KPa deg C-1]
            # http://www.fao.org/docrep/X0490E/x0490e07.htm  Equation 7

        else:
            Psycon = 0.665E-3 * self.var.Psurf
            # psychrometric constant [kPa C-1]
            # http://www.fao.org/docrep/X0490E/x0490e07.htm  Equation 8
            # see http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation

            # calculate vapor pressure
            # Fao 56 Page 36
            # calculate actual vapour pressure
            if returnBool('useHuss'):
                # if specific humidity calculate actual vapour pressure this way
                self.var.EAct = (self.var.Psurf * self.var.huss) / ((0.378 * self.var.huss) + 0.622)
                # http://www.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
                # (self.var.Psurf * self.var.Qair)/0.622
                # old calculation not completely ok
            else:
                # if relative humidity
                self.var.EAct = ESat * self.var.hurs / 100.0
                # longwave radiation balance
            RLN = RNup - self.var.Rsdl
            # RDL is stored on disk as W/m2 but converted in MJ/m2/s in readmeteo.py

        # ************************************************************
        # ***** NET ABSORBED RADIATION *******************************
        # ************************************************************

        if returnBool('albedo'):
            if dateVar['newStart'] or dateVar['newMonth']:  # loading every month a new map
                self.var.albedoLand = self.model.data.to_HRU(data=readnetcdf2('albedoMaps', dateVar['currDate'], useDaily='month',value='albedoLand'), fn=None)  # checked
                self.var.albedoOpenWater = self.model.data.to_HRU(data=readnetcdf2('albedoMaps', dateVar['currDate'], useDaily='month',value='albedoWater'), fn=None)  # checked
            RNA = np.maximum(((1 - self.var.albedoLand) * self.var.Rsds - RLN) / LatHeatVap, 0.0)
            RNAWater = np.maximum(((1 - self.var.albedoOpenWater) * self.var.Rsds - RLN) / LatHeatVap, 0.0)

        else:
            RNA = np.maximum(((1 - self.var.AlbedoCanopy) * self.var.Rsds - RLN) / LatHeatVap, 0.0)
            # net absorbed radiation of reference vegetation canopy [mm/d]
            # RNASoil = np.maximum(((1 - self.var.AlbedoSoil) * self.var.Rsds - RLN) / LatHeatVap, 0.0)
            # net absorbed radiation of bare soil surface
            RNAWater = np.maximum(((1 - self.var.AlbedoWater) * self.var.Rsds - RLN) / LatHeatVap, 0.0)
            # net absorbed radiation of water surface

        VapPressDef = np.maximum(ESat - self.var.EAct, 0.0)
        Delta = ((4098.0 * ESat) / ((self.var.Tavg + 237.3)**2))
        # slope of saturated vapour pressure curve [kPa/deg C]
        # Equation 13 Chapter 3

        # Chapter 2 Equation 6
        windpart = 900 * self.var.Wind / (self.var.Tavg + 273.16)

        denominator = Delta + Psycon *(1 + 0.34 * self.var.Wind)
        numerator1 = Delta / denominator
        numerator2 = Psycon / denominator
        # the 0.408 constant is replace by 1/LatHeatVap see above

        RNAN = RNA * numerator1
        #RNANSoil = RNASoil * numerator1
        RNANWater = RNAWater * numerator1

        EA = windpart * VapPressDef * numerator2

        # Potential evapo(transpi)ration is calculated for two reference surfaces:
        # 1. Reference vegetation canopy
        # 2. Open water surface
        self.var.ETRef = (RNAN + EA) * 0.001
        # potential reference evapotranspiration rate [m/day]  # from mm to m with 0.001
        #self.var.ESRef = RNANSoil + EA
        # potential evaporation rate from a bare soil surface [m/day]
        self.var.EWRef = (RNANWater + EA) * 0.001
        # potential evaporation rate from water surface [m/day]

        # -> here we are at ET0 (see http://www.fao.org/docrep/X0490E/x0490e04.htm#TopOfPage figure 4:)

        #self.var.sumETRef = self.var.sumETRef + self.var.ETRef*1000


        #if dateVar['curr'] ==32:
        #ii=1

        #report(decompress(self.var.sumETRef), "C:\work\output2/sumetref.map")