# -------------------------------------------------------------------------
# Name:        POTENTIAL REFERENCE EVAPO(TRANSPI)RATION
# Purpose:
#
# Author:      PB
#
# Created:     10/01/2017
# Copyright:   (c) PB 2017
# -------------------------------------------------------------------------

from cwatm.management_modules.data_handling import *


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
        self.var = model.subvar
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
        self.var.cropCorrect = self.model.to_subvar(data=self.var.cropCorrect, fn=None)

    def dynamic(self):
        """
        Dynamic part of the potential evaporation module
        Based on Penman Monteith - FAO 56
        """

        wc2_tavg = 0
        wc4_tavg = 0
        wc2_tmin = 0
        wc4_tmin = 0
        wc2_tmax = 0
        wc4_tmax = 0

        ZeroKelvin = 0.0
        if checkOption('TemperatureInKelvin'):
            # if temperature is in Kelvin -> conversion to deg C
            # TODO in initial there could be a check if temperature > 200 -> automatic change to Kelvin
            ZeroKelvin = 273.15

        TMin = readmeteodata('TminMaps',dateVar['currDate'], addZeros=True, zeros=ZeroKelvin, mapsscale = self.model.var.meteomapsscale)
        if self.model.var.meteodown:
            TMin, wc2_tmin, wc4_tmin = self.model.readmeteo_module.downscaling2(TMin, "downscale_wordclim_tmin", wc2_tmin, wc4_tmin, downscale=1)
        else:
            TMin = self.model.readmeteo_module.downscaling2(TMin, "downscale_wordclim_tmin", wc2_tmin, wc4_tmin, downscale=0)

        if Flags['check']: checkmap('TminMaps', "", self.var.Tmin, True, True, self.var.Tmin)

        TMax = readmeteodata('TmaxMaps', dateVar['currDate'], addZeros=True, zeros=ZeroKelvin, mapsscale = self.model.var.meteomapsscale)
        if self.model.var.meteodown:
            TMax, wc2_tmax, wc4_tmax = self.model.readmeteo_module.downscaling2(TMax, "downscale_wordclim_tmin", wc2_tmax, wc4_tmax, downscale=1)
        else:
            TMax = self.model.readmeteo_module.downscaling2(TMax, "downscale_wordclim_tmin", wc2_tmax, wc4_tmax, downscale=0)

        if Flags['check']: checkmap('TmaxMaps', "", TMax, True, True, TMax)

        tzero = 0
        if checkOption('TemperatureInKelvin'):
            tzero = ZeroKelvin

        # average DAILY temperature (even if you are running the model
        # on say an hourly time step) [degrees C]
        Tavg = readmeteodata('TavgMaps',dateVar['currDate'], addZeros=True, zeros = tzero, mapsscale = self.model.var.meteomapsscale)

        if self.model.var.meteodown:
            Tavg, wc2_tavg, wc4_tavg  = self.model.readmeteo_module.downscaling2(Tavg, "downscale_wordclim_tavg", wc2_tavg, wc4_tavg, downscale=1)
        else:
            Tavg  = self.model.readmeteo_module.downscaling2(Tavg, "downscale_wordclim_tavg", wc2_tavg, wc4_tavg, downscale=0)

        if Flags['check']:
            checkmap('TavgMaps', "", Tavg, True, True, Tavg)

        TMax = self.model.to_subvar(data=TMax, fn=None)  # checked
        TMin = self.model.to_subvar(data=TMin, fn=None)  # checked
        Tavg = self.model.to_subvar(data=Tavg, fn=None)  # checked

        if checkOption('TemperatureInKelvin'):
            TMin -= ZeroKelvin
            TMax -= ZeroKelvin
            Tavg -= ZeroKelvin

        ESatmin = 0.6108* np.exp((17.27 * TMin) / (TMin + 237.3))
        ESatmax = 0.6108* np.exp((17.27 * TMax) / (TMax + 237.3))
        ESat = (ESatmin + ESatmax) / 2.0   # [KPa]
        # http://www.fao.org/docrep/X0490E/x0490e07.htm   equation 11/12

        Psurf = readmeteodata('PSurfMaps', dateVar['currDate'], addZeros=True, mapsscale = self.model.var.meteomapsscale)
        Psurf = self.model.readmeteo_module.downscaling2(Psurf)
        # [Pa] to [KPa]
        Psurf = Psurf * 0.001
        Psurf = self.model.to_subvar(data=Psurf, fn=None)  # checked

        if returnBool('useHuss'):
            #self.var.Qair = readnetcdf2('QAirMaps', dateVar['currDate'], addZeros = True, meteo = True)
            Qair = readmeteodata('QAirMaps', dateVar['currDate'], addZeros=True, mapsscale =self.model.var.meteomapsscale)
            # 2 m istantaneous specific humidity[kg / kg]
        else:
            #self.var.Qair = readnetcdf2('RhsMaps', dateVar['currDate'], addZeros = True, meteo = True)
            Qair = readmeteodata('RhsMaps', dateVar['currDate'], addZeros=True, mapsscale =self.model.var.meteomapsscale)
        Qair = self.model.readmeteo_module.downscaling2(Qair)
        Qair = self.model.to_subvar(data=Qair, fn=None)  # checked

        # Fao 56 Page 36
        # calculate actual vapour pressure
        if returnBool('useHuss'):
            # if specific humidity calculate actual vapour pressure this way
            EAct = (Psurf * Qair) / ((0.378 * Qair) + 0.622)
            # http://www.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
            # (self.var.Psurf * self.var.Qair)/0.622
            # old calculation not completely ok
        else:
            # if relative humidity
            EAct = ESat * Qair / 100.0
        del Qair

        # ************************************************************
        # ***** NET ABSORBED RADIATION *******************************
        # ************************************************************
        LatHeatVap = 2.501 - 0.002361 * Tavg
        # latent heat of vaporization [MJ/kg]

        EmNet = (0.34 - 0.14 * np.sqrt(EAct))
        # Net emissivity

        # longwave radiation balance
        RNUp = 4.903E-9 * (((TMin + 273.16) ** 4) + ((TMax + 273.16) ** 4)) / 2
        del TMax
        del TMin

        #Rsds = readnetcdf2('RSDSMaps', dateVar['currDate'], addZeros = True, meteo = True)
        Rsds = readmeteodata('RSDSMaps', dateVar['currDate'], addZeros=True, mapsscale = self.model.var.meteomapsscale)
        Rsds = self.model.readmeteo_module.downscaling2(Rsds)
            # radiation surface downwelling shortwave maps [W/m2]
        #Rsdl = readnetcdf2('RSDLMaps', dateVar['currDate'], addZeros = True, meteo = True)
        Rsdl = readmeteodata('RSDLMaps', dateVar['currDate'], addZeros=True, mapsscale = self.model.var.meteomapsscale)
        Rsdl = self.model.readmeteo_module.downscaling2(Rsdl)
        # Conversion factor from [W] to [MJ]
        WtoMJ = 86400 * 1E-6

        # conversion from W/m2 to MJ/m2/day
        Rsds = Rsds * WtoMJ
        Rsdl = Rsdl * WtoMJ

        Rsdl = self.model.to_subvar(data=Rsdl, fn=None)  # checked
        Rsds = self.model.to_subvar(data=Rsds, fn=None)  # checked

        # Up longwave radiation [MJ/m2/day]
        RLN = RNUp - Rsdl
        # RDL is stored on disk as W/m2 but converted in MJ/m2/s in readmeteo.py

        # TODO: Make albedo dynamic based on land type
        albedoLand = readnetcdf2('albedoMaps', dateVar['currDate'], useDaily='month',value='albedoLand')
        albedoLand = self.model.to_subvar(data=albedoLand, fn=None)  # checked
        albedoOpenWater = readnetcdf2('albedoMaps', dateVar['currDate'], useDaily='month',value='albedoWater')
        albedoOpenWater = self.model.to_subvar(data=albedoOpenWater, fn=None)  # checked
        RNA = np.maximum(((1 - albedoLand) * Rsds - RLN) / LatHeatVap, 0.0)
        RNAWater = np.maximum(((1 - albedoOpenWater) * Rsds - RLN) / LatHeatVap, 0.0)

        VapPressDef = np.maximum(ESat - EAct, 0.0)
        Delta = ((4098.0 * ESat) / ((Tavg + 237.3)**2))
        # slope of saturated vapour pressure curve [mbar/deg C]
        Psycon = 0.665E-3 * Psurf
        del Psurf
        # psychrometric constant [kPa C-1]
        # http://www.fao.org/docrep/ X0490E/ x0490e07.htm  Equation 8
        # see http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation

        # wind speed maps at 10m [m/s]
        Wind = readmeteodata('WindMaps', dateVar['currDate'], addZeros=True, mapsscale = self.model.var.meteomapsscale)
        Wind = self.model.readmeteo_module.downscaling2(Wind)
        Wind = self.model.to_subvar(data=Wind, fn=None)  # checked

        # Adjust wind speed for measurement height: wind speed measured at
        # 10 m, but needed at 2 m height
        # Shuttleworth, W.J. (1993) in Maidment, D.R. (1993), p. 4.36
        Wind = Wind * 0.749

        windpart = 900 * Wind / (Tavg + 273.16)
        denominator = Delta + Psycon *(1 + 0.34 * Wind)
        numerator1 = Delta / denominator
        numerator2 = Psycon / denominator

        del Wind

        RNAN = RNA * numerator1
        #RNANSoil = RNASoil * numerator1
        RNANWater = RNAWater * numerator1

        EA = windpart * VapPressDef * numerator2

        # Potential evapo(transpi)ration is calculated for two reference surfaces:
        # 1. Reference vegetation canopy
        # 2. Open water surface
        ETRef = (RNAN + EA) * 0.001
        # potential reference evapotranspiration rate [m/day]  # from mm to m with 0.001
        #self.var.ESRef = RNANSoil + EA
        # potential evaporation rate from a bare soil surface [m/day]
        EWRef = (RNANWater + EA) * 0.001
        # potential evaporation rate from water surface [m/day]

        # -> here we are at ET0 (see http://www.fao.org/docrep/X0490E/x0490e04.htm#TopOfPage figure 4:)

        #self.var.sumETRef = self.var.sumETRef + self.var.ETRef*1000


        #if dateVar['curr'] ==32:
        #ii=1

        #report(decompress(self.var.sumETRef), "C:\work\output2/sumetref.map")

        return Tavg, ETRef, EWRef