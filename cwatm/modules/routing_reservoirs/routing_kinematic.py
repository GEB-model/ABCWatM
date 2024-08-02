# -------------------------------------------------------------------------
# Name:        Routing module - Kinematic wave
# Purpose:
#
# Author:      PB
#
# Created:     17/01/2017
# Copyright:   (c) PB 2017
# -------------------------------------------------------------------------

from ..routing_reservoirs.routing_sub import (
    define_river_network,
    upstreamArea,
    kinematic,
)
import numpy as np


class routing_kinematic(object):
    """
    ROUTING

    routing using the kinematic wave


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m
    load_initial
    waterbalance_module
    DtSec                 number of seconds per timestep (default = 86400)                                  s
    waterBodyID           lakes/reservoirs map with a single ID for each lake/reservoir                     --
    UpArea1               upstream area of a grid cell                                                      m2
    dirUp                 river network in upstream direction                                               --
    lddCompress           compressed river network (without missing values)                                 --
    dirupLen_LR           number of bifurcation upstream lake/reservoir                                     --
    dirupID_LR            index river upstream lake/reservoir                                               --
    dirDown_LR            river network direktion downstream lake/reservoir                                 --
    lendirDown_LR         number of river network connections lake/reservoir                                --
    compress_LR           boolean map as mask map for compressing lake/reservoir                            --
    lakeArea              area of each lake/reservoir                                                       m2
    lakeEvaFactor         a factor which increases evaporation from lake because of wind                    --
    lakeEvaFactorC        compressed map of a factor which increases evaporation from lake because of wind  --
    EvapWaterBodyM
    lakeResInflowM
    lakeResOutflowM
    dtRouting             number of seconds per routing timestep                                            s
    lakeResStorage
    evapWaterBodyC
    sumLakeEvapWaterBody
    noRoutingSteps
    discharge             discharge                                                                         m3/s
    runoff
    cellArea              Cell area [m²] of each simulated mesh
    downstruct
    prelakeResStorage
    act_SurfaceWaterAbst
    fracVegCover          Fraction of area covered by the corresponding landcover type
    openWaterEvap         Simulated evaporation from open areas                                             m
    catchmentAll
    chanLength
    totalCrossSectionAre
    dirupLen
    dirupID
    catchment
    dirDown
    lendirDown
    UpArea
    beta
    chanMan
    chanGrad
    chanWidth
    chanDepth
    invbeta
    invchanLength
    invdtRouting
    totalCrossSectionAre
    chanWettedPerimeterA
    alpPower
    channelAlpha
    invchannelAlpha
    channelStorageM3
    riverbedExchange
    sumbalance
    prechannelStorageM3
    EvapoChannel
    QDelta
    act_bigLakeResAbst
    act_smallLakeResAbst
    returnFlow
    sumsideflow
    inflowDt
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        """
        Initial part of the routing module

        * load and create a river network
        * calculate river network parameter e.g. river length, width, depth, gradient etc.
        * calculate initial filling
        * calculate manning's roughness coefficient
        """
        self.var = model.data.grid
        self.model = model

        ldd = self.var.load(
            self.model.model_structure["grid"]["routing/kinematic/ldd"],
            compress=False,
        )

        (
            self.var.lddCompress,
            dirshort,
            self.var.dirUp,
            self.var.dirupLen,
            self.var.dirupID,
            self.var.downstruct,
            self.var.catchment,
            self.var.dirDown,
            self.var.lendirDown,
        ) = define_river_network(ldd, self.model.data.grid)

        # self.var.ups = upstreamArea(dirDown, dirshort, self.var.cellArea)
        self.var.UpArea1 = upstreamArea(
            self.var.dirDown, dirshort, self.var.full_compressed(1, dtype=np.float64)
        )
        self.var.UpArea = upstreamArea(
            self.var.dirDown, dirshort, self.var.cellArea.astype(np.float64)
        )

        # ---------------------------------------------------------------
        # Calibration
        # mannings roughness factor 0.1 - 10.0

        # number of substep per day
        self.var.noRoutingSteps = 24
        # kinematic wave parameter: 0.6 is for broad sheet flow
        self.var.beta = 0.6  # TODO: Make this a parameter
        # Channel Manning's n
        self.var.chanMan = (
            self.var.load(
                self.model.model_structure["grid"]["routing/kinematic/mannings"]
            )
            * self.model.config["parameters"]["manningsN"]
        )
        # Channel gradient (fraction, dy/dx)
        minimum_channel_gradient = 0.0001
        self.var.chanGrad = np.maximum(
            self.var.load(
                self.model.model_structure["grid"]["routing/kinematic/channel_slope"]
            ),
            minimum_channel_gradient,
        )
        # Channel length [meters]
        self.var.chanLength = self.var.load(
            self.model.model_structure["grid"]["routing/kinematic/channel_length"]
        )
        # Channel bottom width [meters]
        self.var.chanWidth = self.var.load(
            self.model.model_structure["grid"]["routing/kinematic/channel_width"]
        )

        # Bankfull channel depth [meters]
        self.var.chanDepth = self.var.load(
            self.model.model_structure["grid"]["routing/kinematic/channel_depth"]
        )

        # -----------------------------------------------
        # Inverse of beta for kinematic wave
        self.var.invbeta = 1 / self.var.beta
        # Inverse of channel length [1/m]
        self.var.invchanLength = 1 / self.var.chanLength

        # Corresponding sub-timestep (seconds)
        self.var.dtRouting = self.model.DtSec / self.var.noRoutingSteps
        self.var.invdtRouting = 1 / self.var.dtRouting

        # -----------------------------------------------
        # ***** CHANNEL GEOMETRY  ************************************

        # Area (sq m) of bank full discharge cross section [m2]
        self.var.totalCrossSectionAreaBankFull = self.var.chanDepth * self.var.chanWidth
        # Cross-sectional area at half bankfull [m2]
        # This can be used to initialise channel flow (see below)
        # TotalCrossSectionAreaHalfBankFull = 0.5 * self.var.TotalCrossSectionAreaBankFull
        self.var.totalCrossSectionArea = 0.5 * self.var.totalCrossSectionAreaBankFull
        # Total cross-sectional area [m2]: if initial value in binding equals -9999 the value at half bankfull is used,

        # -----------------------------------------------
        # ***** CHANNEL ALPHA (KIN. WAVE)*****************************
        # ************************************************************
        # Following calculations are needed to calculate Alpha parameter in kinematic
        # wave. Alpha currently fixed at half of bankful depth

        # Reference water depth for calculation of Alpha: half of bankfull
        # chanDepthAlpha = 0.5 * self.var.chanDepth
        # Channel wetted perimeter [m]
        self.var.chanWettedPerimeterAlpha = (
            self.var.chanWidth + 2 * 0.5 * self.var.chanDepth
        )

        # ChannelAlpha for kinematic wave
        alpTermChan = (self.var.chanMan / (np.sqrt(self.var.chanGrad))) ** self.var.beta
        self.var.alpPower = self.var.beta / 1.5
        self.var.channelAlpha = (
            alpTermChan * (self.var.chanWettedPerimeterAlpha**self.var.alpPower) * 2.5
        )
        self.var.invchannelAlpha = 1.0 / self.var.channelAlpha

        # -----------------------------------------------
        # ***** CHANNEL INITIAL DISCHARGE ****************************

        # channel water volume [m3]
        # Initialise water volume in kinematic wave channels [m3]
        channelStorageM3Ini = self.var.totalCrossSectionArea * self.var.chanLength * 0.1
        self.var.channelStorageM3 = self.var.load_initial(
            "channelStorageM3", default=channelStorageM3Ini
        )
        self.var.prechannelStorageM3 = self.var.channelStorageM3.copy()

        # Initialise discharge at kinematic wave pixels (note that InvBeta is
        # simply 1/beta, computational efficiency!)
        # self.var.chanQKin = np.where(self.var.channelAlpha > 0, (self.var.totalCrossSectionArea / self.var.channelAlpha) ** self.var.invbeta, 0.)
        dischargeIni = (
            self.var.channelStorageM3
            * self.var.invchanLength
            * self.var.invchannelAlpha
        ) ** self.var.invbeta
        self.var.discharge = self.var.load_initial("discharge", default=dischargeIni)
        self.var.discharge_substep = self.var.load_initial(
            "discharge_substep",
            default=np.full(
                (self.var.noRoutingSteps, self.var.discharge.size),
                0,
                dtype=self.var.discharge.dtype,
            ),
        )
        # self.var.chanQKin = chanQKinIni

        # self.var.riverbedExchangeM = globals.inZero.copy()
        # self.var.riverbedExchangeM = self.var.load_initial("riverbedExchange", default = globals.inZero.copy())
        # self.var.discharge = self.var.chanQKin.copy()

        # factor for evaporation from lakes, reservoirs and open channels
        self.var.lakeEvaFactor = (
            self.var.full_compressed(0, dtype=np.float32)
            + self.model.config["parameters"]["lakeEvaFactor"]
        )

        if self.model.CHECK_WATER_BALANCE:
            self.var.catchmentAll = self.model.data.grid.full_compressed(0.0)
            self.var.sumbalance = 0

    def step(self, openWaterEvap, channel_abstraction_m, returnFlow):
        """
        Dynamic part of the routing module

        * calculate evaporation from channels
        * calculate riverbed exchange between riverbed and groundwater
        * if option **waterbodies** is true, calculate retention from water bodies
        * calculate sideflow -> inflow to river
        * calculate kinematic wave -> using C++ library for computational speed
        """

        if self.model.CHECK_WATER_BALANCE:
            self.var.prechannelStorageM3 = self.var.channelStorageM3.copy()
            self.var.prelakeResStorage = self.var.storage.copy()

        # Evaporation from open channel
        # from big lakes/res and small lakes/res is calculated separately
        channelFraction = np.minimum(
            1.0, self.var.chanWidth * self.var.chanLength / self.var.cellArea
        )
        # put all the water area in which is not reflected in the lakes ,res
        # channelFraction = np.maximum(self.var.fracVegCover[5], channelFraction)

        EWRefact = self.var.lakeEvaFactor * self.model.data.to_grid(
            HRU_data=self.model.data.HRU.EWRef, fn="weightedmean"
        ) - self.model.data.to_grid(HRU_data=openWaterEvap, fn="weightedmean")
        # evaporation from channel minus the calculated evaporation from rainfall
        EvapoChannel = EWRefact * channelFraction * self.var.cellArea
        # EvapoChannel = self.var.EWRef * channelFraction * self.var.cellArea

        # restrict to 95% of channel storage -> something should stay in the river
        EvapoChannel = np.where(
            (0.95 * self.var.channelStorageM3 - EvapoChannel) > 0.0,
            EvapoChannel,
            0.95 * self.var.channelStorageM3,
        )

        # riverbed infiltration (m3):
        # - current implementation based on Inge's principle (later, will be based on groundater head (MODFLOW) and can be negative)
        # - happening only if 0.0 < baseflow < nonFossilGroundwaterAbs
        # - infiltration rate will be based on aquifer saturated conductivity
        # - limited to fracWat
        # - limited to available channelStorageM3
        # - this infiltration will be handed to groundwater in the next time step

        """
        self.var.riverbedExchange = np.maximum(0.0,  np.minimum(self.var.channelStorageM3, np.where(self.var.baseflow > 0.0, \
                                np.where(self.var.nonFossilGroundwaterAbs > self.var.baseflow, \
                                self.var.kSatAquifer * self.var.fracVegCover[5] * self.var.cellArea, \
                                0.0), 0.0)))
        # to avoid flip flop
        self.var.riverbedExchange = np.minimum(self.var.riverbedExchange, 0.95 * self.var.channelStorageM3)


                if self.var.modflow:
            self.var.interflow[No] = np.where(self.var.capriseindex == 100, toGWorInterflow,
                                              self.var.percolationImp * toGWorInterflow)
        else:
            self.var.interflow[No] = self.var.percolationImp * toGWorInterflow
        """

        # add reservoirs depending on year

        # ------------------------------------------------------------
        # evaporation from water bodies (m3), will be limited by available water in lakes and reservoirs
        # calculate outflow from lakes and reservoirs

        # average evaporation overeach lake
        EWRefavg = (
            np.bincount(self.var.waterBodyID, weights=EWRefact)
            / np.bincount(self.var.waterBodyID)
        )[self.var.waterBodyIDC]
        self.var.evapWaterBodyC = (
            EWRefavg * self.var.lakeAreaC / self.var.noRoutingSteps
        )
        assert np.all(self.var.evapWaterBodyC >= 0.0), "evapWaterBodyC < 0.0"

        if self.model.use_gpu:
            fraction_water = cp.array(self.model.data.HRU.land_use_ratio)
        else:
            fraction_water = np.array(self.model.data.HRU.land_use_ratio)
        fraction_water[self.model.data.HRU.land_use_type != 5] = 0
        fraction_water = self.model.data.to_grid(HRU_data=fraction_water, fn="sum")

        EvapoChannel = np.where(
            self.var.waterBodyID > 0,
            (1 - fraction_water) * EvapoChannel,
            EvapoChannel,
        )
        # self.var.riverbedExchange = np.where(self.var.waterBodyID > 0, 0., self.var.riverbedExchange)

        EvapoChannelM3Dt = EvapoChannel / self.var.noRoutingSteps
        # riverbedExchangeDt = self.var.riverbedExchangeM3 / self.var.noRoutingSteps
        # del self.var.riverbedExchangeM3

        WDAddM3Dt = 0
        # WDAddM3Dt = self.var.act_SurfaceWaterAbstract.copy() #MS CWatM edit Shouldn't this only be from the river abstractions? Currently includes the larger reservoir...
        WDAddMDt = channel_abstraction_m
        # return flow from (m) non irrigation water demand
        # WDAddM3Dt = WDAddM3Dt - self.var.nonIrrReturnFlowFraction * self.var.act_nonIrrDemand
        WDAddMDt = (
            WDAddMDt - returnFlow
        )  # Couldn't this be negative? If return flow is mainly coming from gw? Fine, then more water going in.
        # print('min(WDAddM3Dt)', (min(WDAddM3Dt*9999999)))
        WDAddM3Dt = WDAddMDt * self.var.cellArea / self.var.noRoutingSteps

        # sideflowChanM3 -= self.var.sum_act_SurfaceWaterAbstract * self.var.cellArea
        # return flow from (m) non irrigation water demand
        # self.var.nonIrrReturnFlow = self.var.nonIrrReturnFlowFraction * self.var.nonIrrDemand
        # sideflowChanM3 +=  self.var.nonIrrReturnFlow * self.var.cellArea
        # sideflowChan = sideflowChanM3 * self.var.invchanLength * self.var.invdtRouting

        # ------------------------------------------------------
        # ***** SIDEFLOW **************************************

        runoffM3 = self.var.runoff * self.var.cellArea / self.var.noRoutingSteps

        # ************************************************************
        # ***** KINEMATIC WAVE                        ****************
        # ************************************************************

        self.var.sumsideflow = 0
        # Question: Is this fine?? It is already used above differently
        # self.var.prechannelStorageM3 = self.var.channelAlpha * self.var.chanLength * self.var.discharge ** self.var.beta

        self.var.discharge_substep = np.full(
            (self.var.noRoutingSteps, self.var.discharge.size),
            np.nan,
            dtype=self.var.discharge.dtype,
        )
        pre_channelStorageM3 = self.var.channelStorageM3.copy()
        for subrouting_step in range(self.var.noRoutingSteps):
            # Runoff - Evaporation ( -riverbed exchange), this could be negative  with riverbed exhange also
            sideflowChanM3 = runoffM3.copy()
            # minus evaporation from channels
            sideflowChanM3 -= EvapoChannelM3Dt
            # # minus riverbed exchange
            # sideflowChanM3 -= riverbedExchangeDt

            sideflowChanM3 -= WDAddM3Dt
            # minus waterdemand + returnflow

            outflow_to_river_network = (
                self.model.lakes_reservoirs_module.dynamic_inloop(subrouting_step)
            )
            sideflowChanM3 += outflow_to_river_network

            sideflowChan = sideflowChanM3 * self.var.invchanLength / self.var.dtRouting

            self.var.discharge = kinematic(
                self.var.discharge,
                sideflowChan.astype(np.float32),
                self.var.dirDown_LR,
                self.var.dirupLen_LR,
                self.var.dirupID_LR,
                self.var.channelAlpha,
                self.var.beta,
                self.var.dtRouting,
                self.var.chanLength,
            )

            self.var.discharge_substep[subrouting_step, :] = self.var.discharge.copy()

            self.var.sumsideflow = self.var.sumsideflow + sideflowChanM3

        assert not np.isnan(self.var.discharge).any()

        self.var.channelStorageM3 = (
            self.var.channelAlpha
            * self.var.chanLength
            * self.var.discharge**self.var.beta
        )
