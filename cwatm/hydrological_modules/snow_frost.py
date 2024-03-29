# -------------------------------------------------------------------------
# Name:        Snow module
# Purpose:
#
# Author:      PB
#
# Created:     13/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules.data_handling import *
import numpy as np
import math

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass


class snow_frost(object):
    """
    RAIN AND SNOW

    Domain: snow calculations evaluated for center points of up to 7 sub-pixel
    snow zones 1 -7 which each occupy a part of the pixel surface

    Variables *snow* and *rain* at end of this module are the pixel-average snowfall and rain


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    Tavg                  average air Temperature (input for the model)                                     K
    load_initial
    waterbalance_module
    Rain                  Precipitation less snow                                                           m
    SnowMelt              total snow melt from all layers                                                   m
    SnowCover             snow cover (sum over all layers)                                                  m
    ElevationStD
    Precipitation         Precipitation (input for the model)                                               m
    DtDay                 seconds in a timestep (default=86400)                                             s
    numberSnowLayersFloa
    numberSnowLayers      Number of snow layers (up to 10)                                                  --
    glaciertransportZone  Number of layers which can be mimiced as glacier transport zone                   --
    deltaInvNorm          Quantile of the normal distribution (for different numbers of snow layers)        --
    DeltaTSnow            Temperature lapse rate x std. deviation of elevation                              C°
    SnowDayDegrees        day of the year to degrees: 360/365.25 = 0.9856                                   --
    summerSeasonStart     day when summer season starts = 165                                               --
    IceDayDegrees         days of summer (15th June-15th Sept.) to degree: 180/(259-165)                    --
    SnowSeason            seasonal melt factor                                                              m C°-1 da
    TempSnow              Average temperature at which snow melts                                           C°
    SnowFactor            Multiplier applied to precipitation that falls as snow                            --
    SnowMeltCoef          Snow melt coefficient - default: 0.004                                            --
    IceMeltCoef           Ice melt coefficnet - default  0.007                                              --
    TempMelt              Average temperature at which snow melts                                           C°
    SnowCoverS            snow cover for each layer                                                         m
    Kfrost                Snow depth reduction coefficient, (HH, p. 7.28)                                   m-1
    Afrost                Daily decay coefficient, (Handbook of Hydrology, p. 7.28)                         --
    FrostIndexThreshold   Degree Days Frost Threshold (stops infiltration, percolation and capillary rise)  --
    SnowWaterEquivalent   Snow water equivalent, (based on snow density of 450 kg/m3) (e.g. Tarboton and L  --
    FrostIndex            FrostIndex - Molnau and Bissel (1983), A Continuous Frozen Ground Index for Floo  --
    extfrostindex         Flag for second frostindex                                                        --
    FrostIndexThreshold2  FrostIndex2 - Molnau and Bissel (1983), A Continuous Frozen Ground Index for Flo
    frostInd1             forstindex 1
    frostInd2             frostindex 2
    frostindexS           array for frostindex
    prevSnowCover         snow cover of previous day (only for water balance)                               m
    Snow                  Snow (equal to a part of Precipitation)                                           m
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

    def initial(self, ElevationStD):
        """
        Initial part of the snow and frost module

        * loads all the parameters for the day-degree approach for rain, snow and snowmelt
        * loads the parameter for frost
        """

        self.var.numberSnowLayersFloat = loadmap("NumberSnowLayers")  # default 3
        self.var.numberSnowLayers = int(self.var.numberSnowLayersFloat)
        self.var.glaciertransportZone = int(
            loadmap("GlacierTransportZone")
        )  # default 1 -> highest zone is transported to middle zone

        # Difference between (average) air temperature at average elevation of
        # pixel and centers of upper- and lower elevation zones [deg C]
        # ElevationStD:   Standard Deviation of the DEM
        # 0.9674:    Quantile of the normal distribution: u(0,833)=0.9674 to split the pixel in 3 equal parts.
        # for different number of layers
        #  Number: 2 ,3, 4, 5, 6, 7, ,8, 9, 10
        dn = {}
        dn[1] = np.array([0])
        dn[2] = np.array([-0.67448975, 0.67448975])
        dn[3] = np.array([-0.96742157, 0.0, 0.96742157])
        dn[5] = np.array([-1.28155157, -0.52440051, 0.0, 0.52440051, 1.28155157])
        dn[7] = np.array(
            [
                -1.46523379,
                -0.79163861,
                -0.36610636,
                0.0,
                0.36610636,
                0.79163861,
                1.46523379,
            ]
        )
        dn[9] = np.array(
            [
                -1.59321882,
                -0.96742157,
                -0.5894558,
                -0.28221615,
                0.0,
                0.28221615,
                0.5894558,
                0.96742157,
                1.59321882,
            ]
        )
        dn[10] = np.array(
            [
                -1.64485363,
                -1.03643339,
                -0.67448975,
                -0.38532047,
                -0.12566135,
                0.12566135,
                0.38532047,
                0.67448975,
                1.03643339,
                1.64485363,
            ]
        )

        # divNo = 1./float(self.var.numberSnowLayers)
        # deltaNorm = np.linspace(divNo/2, 1-divNo/2, self.var.numberSnowLayers)
        # self.var.deltaInvNorm = norm.ppf(deltaNorm)
        self.var.deltaInvNorm = dn[self.var.numberSnowLayers]

        # self.var.DeltaTSnow =  uNorm[self.var.numberSnowLayers] * ElevationStD * loadmap('TemperatureLapseRate')
        # self.var.DeltaTSnow = 0.9674 * ElevationStD * loadmap('TemperatureLapseRate')
        self.var.DeltaTSnow = ElevationStD * self.model.data.to_HRU(
            data=loadmap("TemperatureLapseRate"), fn=None
        )  # checked

        self.var.SnowDayDegrees = 0.9856
        # day of the year to degrees: 360/365.25 = 0.9856
        self.var.summerSeasonStart = 165
        # self.var.IceDayDegrees = 1.915
        self.var.IceDayDegrees = 180.0 / (259 - self.var.summerSeasonStart)
        # days of summer (15th June-15th Sept.) to degree: 180/(259-165)
        self.var.SnowSeason = loadmap("SnowSeasonAdj") * 0.5
        # default value of range  of seasonal melt factor is set to 0.001 m C-1 day-1
        # 0.5 x range of sinus function [-1,1]
        self.var.TempSnow = loadmap("TempSnow")
        self.var.SnowFactor = loadmap("SnowFactor")
        self.var.SnowMeltCoef = loadmap("SnowMeltCoef")
        self.var.IceMeltCoef = loadmap("IceMeltCoef")

        self.var.TempMelt = loadmap("TempMelt")

        # initialize snowcovers as many as snow layers -> read them as SnowCover1 , SnowCover2 ...
        # SnowCover1 is the highest zone
        SnowCoverS = np.tile(
            self.model.data.to_HRU(
                data=self.model.data.grid.full_compressed(0, dtype=np.float32), fn=None
            ),
            (self.var.numberSnowLayers, 1),
        )
        self.var.SnowCoverS = self.model.data.HRU.load_initial(
            "SnowCoverS", default=SnowCoverS
        )

        # Pixel-average initial snow cover: average of values in 3 elevation
        # zones

        # ---------------------------------------------------------------------------------
        # Initial part of frost index

        # self.var.Kfrost = loadmap('Kfrost')
        self.var.Afrost = loadmap("Afrost")
        self.var.FrostIndexThreshold = loadmap("FrostIndexThreshold")
        self.var.SnowWaterEquivalent = loadmap("SnowWaterEquivalent")

        self.var.FrostIndex = self.model.data.HRU.load_initial(
            "FrostIndex",
            default=self.model.data.HRU.full_compressed(0, dtype=np.float32),
        )

        self.var.extfrostindex = False
        if "morefrost" in binding:
            self.var.extfrostindex = returnBool("morefrost")
            self.var.FrostIndexThreshold2 = loadmap("FrostIndexThreshold2")
            self.var.frostInd1 = globals.inZero
            self.var.frostInd2 = globals.inZero
            self.var.frostindexS = []
            for i in range(self.var.numberSnowLayers):
                self.var.frostindexS.append(globals.inZero)

    def dynamic(self):
        """
        Dynamic part of the snow module

        Distinguish between rain/snow and calculates snow melt and glacier melt
        The equation is a modification of:

        References:
            Speers, D.D., Versteeg, J.D. (1979) Runoff forecasting for reservoir operations - the pastand the future. In: Proceedings 52nd Western Snow Conference, 149-156

        Frost index in soil [degree days] based on:

        References:
            Molnau and Bissel (1983, A Continuous Frozen Ground Index for Flood Forecasting. In: Maidment, Handbook of Hydrology, p. 7.28, 7.55)

        Todo:
            calculate sinus shape function for the southern hemisspere
        """
        if checkOption("calcWaterBalance"):
            self.var.prevSnowCover = np.sum(self.var.SnowCoverS, axis=0)

        day_of_year = self.model.current_time.timetuple().tm_yday
        SeasSnowMeltCoef = (
            self.var.SnowSeason
            * np.sin(math.radians((day_of_year - 81) * self.var.SnowDayDegrees))
            + self.var.SnowMeltCoef
        )

        # SeasSnowMeltCoef = SnowSeason * sin((dateVar['doy']-81)* SnowDayDegrees) + SnowMeltCoef;

        # sinus shaped function between the
        # annual minimum (December 21st) and annual maximum (June 21st)
        # TODO change this for the southern hemisspere

        if (day_of_year > self.var.summerSeasonStart) and (day_of_year < 260):
            SummerSeason = np.sin(
                math.radians(
                    (day_of_year - self.var.summerSeasonStart) * self.var.IceDayDegrees
                )
            )
        else:
            SummerSeason = 0.0

        Snow = self.var.full_compressed(0, dtype=np.float32)
        self.var.Rain = self.var.full_compressed(0, dtype=np.float32)
        self.var.SnowMelt = self.var.full_compressed(0, dtype=np.float32)

        tas_C = self.var.tas - 273.15
        self.var.precipitation_m_day = (
            self.model.DtDay * 0.001 * 86400.0 * self.var.pr  # kg/m2/s to m/day
        )

        for i in range(self.var.numberSnowLayers):
            TavgS = tas_C + self.var.DeltaTSnow * self.var.deltaInvNorm[i]
            # Temperature at center of each zone (temperature at zone B equals Tavg)
            # i=0 -> highest zone
            # i=2 -> lower zone
            SnowS = np.where(
                TavgS < self.var.TempSnow,
                self.var.SnowFactor * self.var.precipitation_m_day,
                self.var.full_compressed(0, dtype=np.float32),
            )
            # Precipitation is assumed to be snow if daily average temperature is below TempSnow
            # Snow is multiplied by correction factor to account for undercatch of
            # snow precipitation (which is common)
            RainS = np.where(
                TavgS >= self.var.TempSnow,
                self.var.precipitation_m_day,
                self.var.full_compressed(0, dtype=np.float32),
            )
            # if it's snowing then no rain
            # snowmelt coeff in m/deg C/day
            SnowMeltS = (
                (TavgS - self.var.TempMelt)
                * SeasSnowMeltCoef
                * (1 + 0.01 * RainS)
                * self.model.DtDay
            )
            SnowMeltS = np.maximum(
                SnowMeltS, self.var.full_compressed(0, dtype=np.float32)
            )

            # for which layer the ice melt is calcultated with the middle temp.
            # for the others it is calculated with the corrected temp
            # this is to mimic glacier transport to lower zones
            if i <= self.var.glaciertransportZone:
                IceMeltS = (
                    tas_C * self.var.IceMeltCoef * self.model.DtDay * SummerSeason
                )
                # if i = 0 and 1 -> higher and middle zone
                # Ice melt coeff in m/C/deg
            else:
                IceMeltS = (
                    TavgS * self.var.IceMeltCoef * self.model.DtDay * SummerSeason
                )

            IceMeltS = np.maximum(
                IceMeltS, self.var.full_compressed(0, dtype=np.float32)
            )
            SnowMeltS = np.maximum(
                np.minimum(SnowMeltS + IceMeltS, self.var.SnowCoverS[i]),
                self.var.full_compressed(0, dtype=np.float32),
            )
            # check if snow+ice not bigger than snowcover
            self.var.SnowCoverS[i] = self.var.SnowCoverS[i] + SnowS - SnowMeltS
            Snow += SnowS
            self.var.Rain += RainS
            self.var.SnowMelt += SnowMeltS

            if self.var.extfrostindex:
                Kfrost = np.where(TavgS < 0, 0.08, 0.5)
                FrostIndexChangeRate = -(1 - self.var.Afrost) * self.var.frostindexS[
                    i
                ] - TavgS * np.exp(
                    -0.4
                    * 100
                    * Kfrost
                    * np.minimum(
                        1.0, self.var.SnowCoverS[i] / self.var.SnowWaterEquivalent
                    )
                )
                self.var.frostindexS[i] = np.maximum(
                    self.var.frostindexS[i] + FrostIndexChangeRate * self.model.DtDay, 0
                )

        if self.var.extfrostindex:
            if dateVar["curr"] >= dateVar["intSpin"]:
                for i in range(self.var.numberSnowLayers):
                    self.var.frostInd1 = np.where(
                        self.var.frostindexS[i] > self.var.FrostIndexThreshold,
                        self.var.frostInd1 + 1 / float(self.var.numberSnowLayers),
                        self.var.frostInd1,
                    )
                    self.var.frostInd2 = np.where(
                        self.var.frostindexS[i] > self.var.FrostIndexThreshold2,
                        self.var.frostInd2 + 1 / float(self.var.numberSnowLayers),
                        self.var.frostInd2,
                    )
            if dateVar["currDate"] == dateVar["dateEnd"]:
                self.var.frostInd1 = self.var.frostInd1 / float(dateVar["diffdays"])
                self.var.frostInd2 = self.var.frostInd2 / float(dateVar["diffdays"])

        Snow /= self.var.numberSnowLayersFloat
        self.var.Rain /= self.var.numberSnowLayersFloat
        self.var.SnowMelt /= self.var.numberSnowLayersFloat
        # all in pixel

        # DEBUG Snow
        # if checkOption('calcWaterBalance'):
        #     self.var.waterbalance_module.waterBalanceCheck(
        #         [Snow],  # In
        #         [self.var.SnowMelt],  # Out
        #         [self.var.prevSnowCover],   # prev storage
        #         [np.sum(self.var.SnowCoverS, axis=0) / self.var.numberSnowLayersFloat],
        #         "Snow1", False)

        # ---------------------------------------------------------------------------------
        # Dynamic part of frost index
        Kfrost = np.where(tas_C < 0, 0.08, 0.5).astype(tas_C.dtype)
        FrostIndexChangeRate = -(
            1 - self.var.Afrost
        ) * self.var.FrostIndex - tas_C * np.exp(
            -0.4
            * 100
            * Kfrost
            * np.minimum(
                1.0,
                (np.sum(self.var.SnowCoverS, axis=0) / self.var.numberSnowLayersFloat)
                / self.var.SnowWaterEquivalent,
            )
        )
        # Rate of change of frost index (expressed as rate, [degree days/day])
        self.var.FrostIndex = np.maximum(
            self.var.FrostIndex + FrostIndexChangeRate * self.model.DtDay, 0
        )
        # frost index in soil [degree days] based on Molnau and Bissel (1983, A Continuous Frozen Ground Index for Flood
        # Forecasting. In: Maidment, Handbook of Hydrology, p. 7.28, 7.55)
        # if Tavg is above zero, FrostIndex will stay 0
        # if Tavg is negative, FrostIndex will increase with 1 per degree C per day
        # Exponent of 0.04 (instead of 0.4 in HoH): conversion [cm] to [mm]!  -> from cm to m HERE -> 100 * 0.4
        # maximum snowlayer = 1.0 m
        # Division by SnowDensity because SnowDepth is expressed as equivalent water depth(always less than depth of snow pack)
        # SnowWaterEquivalent taken as 0.45
        # Afrost, (daily decay coefficient) is taken as 0.97 (Handbook of Hydrology, p. 7.28)
        # Kfrost, (snow depth reduction coefficient) is taken as 0.57 [1/cm], (HH, p. 7.28) -> from Molnau taken as 0.5 for t> 0 and 0.08 for T<0

        """
        if self.var.extfrostindex:

            if dateVar['curr'] >= dateVar['intSpin']:
                self.var.frostInd1 = np.where(self.var.FrostIndex > self.var.FrostIndexThreshold, self.var.frostInd1  +1., self.var.frostInd1)
                self.var.frostInd2 = np.where(self.var.FrostIndex > 84., self.var.frostInd2  +1., self.var.frostInd2)

            if dateVar['currDate'] == dateVar['dateEnd']:
                self.var.frostInd1 = self.var.frostInd1 / float(dateVar['diffdays'])
                self.var.frostInd2 = self.var.frostInd2 / float(dateVar['diffdays'])
                ii = 1
        """
