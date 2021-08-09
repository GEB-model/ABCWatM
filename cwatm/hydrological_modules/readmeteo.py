# -------------------------------------------------------------------------
# Name:        READ METEO input maps
# Purpose:
#
# Author:      PB
#
# Created:     13/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import os
from cwatm.management_modules.data_handling import readmeteodata, cbinding, readCoordNetCDF, divideValues
from cwatm.management_modules.data_handling import *
import scipy.ndimage

class readmeteo(object):
    """
    READ METEOROLOGICAL DATA

    reads all meteorological data from netcdf4 files


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    modflow               Flag: True if modflow_coupling = True in settings file                            --       
    TMin                  minimum air temperature                                                           K        
    TMax                  maximum air temperature                                                           K        
    Psurf                 Instantaneous surface pressure                                                    Pa       
    Qair                  specific humidity                                                                 kg/kg    
    Tavg                  average air Temperature (input for the model)                                     K        
    Rsdl                  long wave downward surface radiation fluxes                                       W/m2     
    Rsds                  short wave downward surface radiation fluxes                                      W/m2     
    Wind                  wind speed                                                                        m/s      
    ETRef                 potential evapotranspiration rate from reference crop                             m        
    EWRef                 potential evaporation rate from water surface                                     m        
    Precipitation         Precipitation (input for the model)                                               m        
    DtDay                 seconds in a timestep (default=86400)                                             s        
    con_precipitation     conversion factor for precipitation                                               --       
    con_e                 conversion factor for evaporation                                                 --       
    meteomapsscale        if meteo maps have the same extend as the other spatial static maps -> meteomaps  --       
    meteodown             if meteo maps should be downscaled                                                --       
    preMaps               choose between steady state precipitation maps for steady state modflow or norma  --       
    tempMaps              choose between steady state temperature maps for steady state modflow or normal   --       
    evaTMaps              choose between steady state ETP water maps for steady state modflow or normal ma  --       
    eva0Maps              choose between steady state ETP reference maps for steady state modflow or norma  --       
    wc2_tavg              High resolution WorldClim map for average temperature                             K        
    wc4_tavg              upscaled to low resolution WorldClim map for average temperature                  K        
    wc2_tmin              High resolution WorldClim map for min temperature                                 K        
    wc4_tmin              upscaled to low resolution WorldClim map for min temperature                      K        
    wc2_tmax              High resolution WorldClim map for max temperature                                 K        
    wc4_tmax              upscaled to low resolution WorldClim map for max temperature                      K        
    wc2_prec              High resolution WorldClim map for precipitation                                   m        
    wc4_prec              upscaled to low resolution WorldClim map for precipitation                        m        
    demAnomaly            digital elevation model anomaly (high resolution - low resolution)                m        
    demHigh               digital elevation model high resolution                                           m        
    prec                  precipitation in m                                                                m        
    temp                  average temperature in Celsius deg                                                C°       
    Tmin                  minimum temperature in Celsius deg                                                C°       
    Tmax                  maximum temperature in celsius deg                                                C°       
    WtoMJ                 Conversion factor from [W] to [MJ] for radiation: 86400 * 1E-6                    --       
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.model = model
        self.var = model.var

    def initial(self):
        """
        Initial part of meteo

        read multiple file of input
        """

        # fit meteorological forcing data to size and resolution of mask map
        #-------------------------------------------------------------------

        name = cbinding('PrecipitationMaps')
        nameall = glob.glob(os.path.normpath(name))
        if not nameall:
            raise CWATMFileError(name, sname='PrecipitationMaps')
        namemeteo = nameall[0]
        latmeteo, lonmeteo, cell, invcellmeteo = readCoordNetCDF(namemeteo)

        nameldd = cbinding('Ldd')
        #nameldd = os.path.splitext(nameldd)[0] + '.nc'
        #latldd, lonldd, cell, invcellldd = readCoordNetCDF(nameldd)
        latldd, lonldd, cell, invcellldd = readCoord(nameldd)
        maskmapAttr['reso_mask_meteo'] = round(invcellldd / invcellmeteo)

        # if meteo maps have the same extend as the other spatial static maps -> meteomapsscale = True
        self.var.meteomapsscale = True
        if invcellmeteo != invcellldd:
            if (not(Flags['quiet'])) and (not(Flags['veryquiet'])) and (not(Flags['check'])):
                msg = "Resolution of meteo forcing is " + str(maskmapAttr['reso_mask_meteo']) + " times higher than base maps."
                print(msg)
            self.var.meteomapsscale = False

        cutmap[0], cutmap[1], cutmap[2], cutmap[3] = mapattrNetCDF(nameldd)
        for i in range(4): cutmapFine[i] = cutmap[i]

        # for downscaling meteomaps , Wordclim data at a finer resolution is used
        # here it is necessary to clip the wordclim data so that they fit to meteo dataset
        self.var.meteodown = False
        if "usemeteodownscaling" in binding:
            self.var.meteodown = returnBool('usemeteodownscaling')

        if self.var.meteodown:
            check_clim = checkMeteo_Wordclim(namemeteo, cbinding('downscale_wordclim_prec'))

        # in case other mapsets are used e.g. Cordex RCM meteo data
        if (latldd != latmeteo) or (lonldd != lonmeteo):
            cutmapFine[0], cutmapFine[1], cutmapFine[2], cutmapFine[3], cutmapVfine[0], cutmapVfine[1], cutmapVfine[2], cutmapVfine[3] = mapattrNetCDFMeteo(namemeteo)

        if not self.var.meteomapsscale:
            # if the cellsize of the spatial dataset e.g. ldd, soil etc is not the same as the meteo maps than:
            cutmapFine[0], cutmapFine[1],cutmapFine[2],cutmapFine[3],cutmapVfine[0], cutmapVfine[1],cutmapVfine[2],cutmapVfine[3]  = mapattrNetCDFMeteo(namemeteo)
            # downscaling wordlclim maps
            for i in range(4): cutmapGlobal[i] = cutmapFine[i]

            if not(check_clim):
               # for downscaling it is always cut from the global map
                if (latldd != latmeteo) or (lonldd != lonmeteo):
                    cutmapGlobal[0] = int(cutmap[0] / maskmapAttr['reso_mask_meteo'])
                    cutmapGlobal[2] = int(cutmap[2] / maskmapAttr['reso_mask_meteo'])
                    cutmapGlobal[1] = int(cutmap[1] / maskmapAttr['reso_mask_meteo']+0.999)
                    cutmapGlobal[3] = int(cutmap[3] / maskmapAttr['reso_mask_meteo']+0.999)

        # -------------------------------------------------------------------



        # test if ModFlow is in the settingsfile
        # if not, use default without Modflow
        self.model.modflow = False
        if "modflow_coupling" in option:
            self.model.modflow = checkOption('modflow_coupling')

        meteomaps = ['PrecipitationMaps', 'TavgMaps','TminMaps','TmaxMaps','PSurfMaps','WindMaps','RSDSMaps','RSDLMaps']
        if returnBool('useHuss'):
            meteomaps.append('QAirMaps')
        else:
            meteomaps.append('RhsMaps')

        multinetdf(meteomaps)

        # downscaling to wordclim, set parameter to 0 in case they are only used as dummy
        self.var.wc2_prec = 0
        self.var.wc4_prec = 0


        # read dem for making a anomolydem between high resolution dem and low resoultion dem

        """
        # for downscaling1
        dem = loadmap('Elevation', compress = False, cut = False)
        demHigh = dem[cutmapFine[2]*6:cutmapFine[3]*6, cutmapFine[0]*6:cutmapFine[1]*6]
        rows = demHigh.shape[0]
        cols = demHigh.shape[1]
        dem2 = demHigh.reshape(rows/6,6,cols/6,6)
        dem3 = np.average(dem2, axis=(1, 3))
        demLow = np.kron(dem3, np.ones((6, 6)))

        demAnomaly = demHigh - demLow
        self.var.demHigh = compressArray(demHigh[cutmapVfine[2]:cutmapVfine[3], cutmapVfine[0]:cutmapVfine[1]],pcr = False)
        self.var.demAnomaly = compressArray(demAnomaly[cutmapVfine[2]:cutmapVfine[3], cutmapVfine[0]:cutmapVfine[1]],pcr = False)
        """

        self.model.subvar.Precipitation = self.model.subvar.full_compressed(0, dtype=np.float32)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

    def downscaling1(self,input, downscale = 0):
        """
        Downscaling based on elevation correction for temperature and pressure

        :param input:
        :param downscale: 0 for no change, 1: for temperature change 6 deg per 1km , 2 for psurf
        :return: input - downscaled input data

        """

        # if meteo maps have the same extend as the other spatial static maps -> meteomapsscale = True
        if not self.var.meteomapsscale:
            down1 = np.kron(input, np.ones((6, 6)))
            down2 = down1[cutmapVfine[2]:cutmapVfine[3], cutmapVfine[0]:cutmapVfine[1]].astype(np.float64)
            down3 = compressArray(down2)
            if downscale == 0:
                input = down3

            if downscale == 1:
                # temperature scaling 6 deg per 1000m difference in altitude
                # see overview in Minder et al 2010 - http://onlinelibrary.wiley.com/doi/10.1029/2009JD013493/full
                tempdiff = -0.006 * self.var.demAnomaly
                input = down3 + tempdiff
            if downscale == 2:
                # psurf correction
                # https://www.sandhurstweather.org.uk/barometric.pdf
                # factor = exp(-elevation / (Temp x 29.263)  Temp in deg K
                demLow = self.var.demHigh - self.var.demAnomaly
                tavgK = self.var.Tavg + 273.15
                factor1 = np.exp(-1 * demLow / (tavgK * 29.263))
                factor2 = np.exp(-1 * self.var.demHigh / (tavgK * 29.263))
                sealevelpressure = down3 / factor1
                input = sealevelpressure * factor2
        return input


    def downscaling2(self,input, downscaleName = "", wc2 = 0 , wc4 = 0, downscale = 0):
        """
        Downscaling based on Delta method:

        Note:

            | **References**
            | Moreno and Hasenauer  2015:
            | ftp://palantir.boku.ac.at/Public/ClimateData/Moreno_et_al-2015-International_Journal_of_Climatology.pdf
            | Mosier et al. 2018:
            | http://onlinelibrary.wiley.com/doi/10.1002/joc.5213/epdf\

        :param input: low input map
        :param downscaleName: High resolution monthly map from WorldClim
        :param wc2: High resolution WorldClim map
        :param wc4: upscaled to low resolution
        :param downscale: 0 for no change, 1: for temperature , 2 for pprecipitation, 3 for psurf
        :return: input - downscaled input data
        :return: wc2
        :return: wc4
        """
        reso = maskmapAttr['reso_mask_meteo']
        resoint = int(reso)

        if self.var.meteomapsscale:
            if downscale == 0:
                return input
            else:
                return input, wc2, wc4


        down3 = np.kron(input, np.ones((resoint, resoint), dtype=input.dtype))
        if downscale == 0:
            down2 = down3[cutmapVfine[2]:cutmapVfine[3], cutmapVfine[0]:cutmapVfine[1]].astype(input.dtype)
            input = compressArray(down2)
            return input
        else:
            if dateVar['newStart'] or dateVar['newMonth']:  # loading every month a new map
                wc1 = readnetcdf2(downscaleName, dateVar['currDate'], useDaily='month', compress = False, cut = False)
                wc2 = wc1[cutmapGlobal[2]*resoint:cutmapGlobal[3]*resoint, cutmapGlobal[0]*resoint:cutmapGlobal[1]*resoint]
                #wc2 = wc1[cutmapGlobal[2] * resoint:cutmapGlobal[3] * resoint, cutmapGlobal[0] * resoint:cutmapGlobal[1] * resoint]
                rows = wc2.shape[0]
                cols = wc2.shape[1]
                wc3 =  wc2.reshape(rows//resoint,resoint,cols//resoint,resoint)
                wc4 =  np.nanmean(wc3, axis=(1, 3))

        if downscale == 1: # Temperature
            diff_wc = wc4 - input
            #diff_wc[np.isnan( diff_wc)] = 0.0
            # could also use np.kron !
            diffSmooth = scipy.ndimage.zoom(diff_wc, resoint, order=1)
            down1 = wc2 - diffSmooth
            down1 = np.where(np.isnan(down1),down3,down1)
        if downscale == 2:  # precipitation
            quot_wc = divideValues(input, wc4)
            quotSmooth = scipy.ndimage.zoom(quot_wc, resoint, order=1)
            down1 = wc2 * quotSmooth
            down1 = np.where(np.isnan(down1),down3,down1)
            down1 = np.where(np.isinf(down1), down3, down1)


        down2 = down1[cutmapVfine[2]:cutmapVfine[3], cutmapVfine[0]:cutmapVfine[1]]
        input = compressArray(down2)
        return input, wc2, wc4

     # --- end downscaling ----------------------------



    def dynamic(self):
        """
        Dynamic part of the readmeteo module

        Read meteo input maps from netcdf files

        Note:
            If option *calc_evaporation* is False only precipitation, avg. temp., and 2 evaporation vlaues are read
            Otherwise all the variable needed for Penman-Monteith

        Note:
            If option *TemperatureInKelvin* = True temperature is assumed to be Kelvin instead of Celsius!

        """

        self.var.Precipitation = readmeteodata('PrecipitationMaps', dateVar['currDate'], addZeros=True, mapsscale = self.var.meteomapsscale) * self.model.DtDay * self.var.con_precipitation
        self.var.Precipitation = np.maximum(0., self.var.Precipitation)

        if self.var.meteodown:
            self.var.Precipitation, self.var.wc2_prec, self.var.wc4_prec = self.downscaling2(self.var.Precipitation, "downscale_wordclim_prec", self.var.wc2_prec, self.var.wc4_prec, downscale=2)
        else:
            self.var.Precipitation = self.downscaling2(self.var.Precipitation, "downscale_wordclim_prec", self.var.wc2_prec, self.var.wc4_prec, downscale=0)

        #self.var.Precipitation = self.var.Precipitation * 1000

        self.var.prec = self.var.Precipitation / self.var.con_precipitation
        # precipitation (conversion to [m] per time step)  `
        if Flags['check']:
            checkmap('PrecipitationMaps', "", self.var.Precipitation, True, True, self.var.Precipitation)

        #self.var.Tavg = readnetcdf2('TavgMaps', dateVar['currDate'], addZeros = True, zeros = ZeroKelvin, meteo = True)


        #self.var.Tavg = downscaling(self.var.Tavg, downscale = 0)

        # -----------------------------------------------------------------------
        # if evaporation has to be calculated load all the meteo map sets
        # Temparture min, max;  Windspeed,  specific humidity or relative humidity
        # psurf, radiation
        # -----------------------------------------------------------------------


            # radiation surface downwelling longwave maps [W/m2]
            #


        #--------------------------------------------------------
        # conversions

        self.model.to_subvar(varname="Precipitation", fn=None)  # checked
