# -------------------------------------------------------------------------
# Name:        Lakes and reservoirs module
# Purpose:
#
# Author:      PB
#
# Created:     01/08/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------
import pandas as pd
import numpy as np

from cwatm.management_modules.data_handling import (
    checkOption,
    loadmap,
    decompress,
    compressArray,
    returnBool,
)
from cwatm.management_modules.globals import binding
from cwatm.hydrological_modules.routing_reservoirs.routing_sub import (
    subcatchment1,
    defLdd2,
    upstream1,
)


def laketotal(values, areaclass):
    """
    numpy area total procedure

    :param values:
    :param areaclass:
    :return: calculates the total area of a class
    """
    return np.take(np.bincount(areaclass, weights=values), areaclass)


def npareamaximum(values, areaclass):
    """
    numpy area maximum procedure

    :param values:
    :param areaclass:
    :return: calculates the maximum of an area of a class
    """
    valueMax = np.zeros(areaclass.max().item() + 1)
    np.maximum.at(valueMax, areaclass, values)
    return np.take(valueMax, areaclass)


class lakes_reservoirs(object):
    r"""
    LAKES AND RESERVOIRS

    Note:

        Calculate water retention in lakes and reservoirs

        Using the **Modified Puls approach** to calculate retention of a lake
        See also: LISFLOOD manual Annex 3 (Burek et al. 2013)

        for Modified Puls Method the Q(inflow)1 has to be used. It is assumed that this is the same as Q(inflow)2 for the first timestep
        has to be checked if this works in forecasting mode!

        Lake Routine using Modified Puls Method (see Maniak, p.331ff)

        .. math::
             {Qin1 + Qin2 \over{2}} - {Qout1 + Qout2 \over{2}} = {S2 - S1 \over{\delta time}}

        changed into:

        .. math::
             {S2 \over{time + Qout2/2}} = {S1 \over{dtime + Qout1/2}} - Qout1 + {Qin1 + Qin2 \over{2}}

        Outgoing discharge (Qout) are linked to storage (S) by elevation.

        Now some assumption to make life easier:

        1.) storage volume is increase proportional to elevation: S = A * H where: H: elevation, A: area of lake

        2.) :math:`Q_{\mathrm{out}} = c * b * H^{2.0}` (c: weir constant, b: width)

             2.0 because it fits to a parabolic cross section see (Aigner 2008) (and it is much easier to calculate (that's the main reason)

        c: for a perfect weir with mu=0.577 and Poleni: :math:`{2 \over{3}} \mu * \sqrt{2*g} = 1.7`

        c: for a parabolic weir: around 1.8

        because it is a imperfect weir: :math:`C = c * 0.85 = 1.5`

        results in formular: :math:`Q = 1.5 * b * H^2 = a*H^2 -> H = \sqrt{Q/a}`

        Solving the equation:

        :math:`{S2 \over{dtime + Qout2/2}} = {S1 \over{dtime + Qout1/2}} - Qout1 + {Qin1 + Qin2 \over{2}}`

        :math:`SI = {S2 \over{dtime}} + {Qout2 \over{2}} = {A*H \over{DtRouting}} + {Q \over{2}} = {A \over{DtRouting*\sqrt{a}* \sqrt{Q}}} + {Q \over{2}}`

        -> replacement: :math:`{A \over{DtSec * \sqrt{a}}} = Lakefactor, Y = \sqrt{Q}`

        :math:`Y^2 + 2 * Lakefactor *Y - 2 * SI=0`

        solution of this quadratic equation:

        :math:`Q = (-LakeFactor + \sqrt{LakeFactor^2+2*SI})^2`


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    load_initial
    saveInit              Flag: if true initial conditions are saved                                        --
    waterBodyID           lakes/reservoirs map with a single ID for each lake/reservoir                     --
    waterBodyOut          biggest outlet (biggest accumulation of ldd network) of a waterbody               --
    dirUp                 river network in upstream direction                                               --
    ldd_LR                change river network (put pits in where lakes are)                                --
    lddCompress_LR        compressed river network lakes/reservoirs (without missing values)                --
    dirUp_LR              river network direction upstream lake/reservoirs                                  --
    dirupLen_LR           number of bifurcation upstream lake/reservoir                                     --
    dirupID_LR            index river upstream lake/reservoir                                               --
    downstruct_LR         river network downstream lake/reservoir                                           --
    catchment_LR          catchments lake/reservoir                                                         --
    dirDown_LR            river network direktion downstream lake/reservoir                                 --
    lendirDown_LR         number of river network connections lake/reservoir                                --
    compress_LR           boolean map as mask map for compressing lake/reservoir                            --
    decompress_LR         boolean map as mask map for decompressing lake/reservoir                          --
    waterBodyOutC         compressed map biggest outlet of each lake/reservoir                              --
    resYearC              compressed map of the year when the reservoirs is operating                       --
    waterBodyTypC         water body types 3 reservoirs and lakes (used as reservoirs but before the year   --
    lakeArea              area of each lake/reservoir                                                       m2
    lakeAreaC             compressed map of the area of each lake/reservoir                                 m2
    lakeDis0              compressed map average discharge at the outlet of a lake/reservoir                m3 s-1
    lakeDis0C             average discharge at the outlet of a lake/reservoir                               m3 s-1
    lakeAC                compressed map of parameter of channel width, gravity and weir coefficient        --
    resVolumeC            compressed map of reservoir volume                                                Million m
    lakeEvaFactorC        compressed map of a factor which increases evaporation from lake because of wind  --
    reslakeoutflow                                                                                          m
    lakeVolume            volume of lakes                                                                   m3
    outLake               outflow from lakes                                                                m
    lakeStorage
    lakeInflow
    lakeOutflow
    reservoirStorage
    MtoM3C                conversion factor from m to m3 (compressed map)                                   --
    EvapWaterBodyM
    lakeResInflowM
    lakeResOutflowM
    lakedaycorrect
    lakeFactor            factor for the Modified Puls approach to calculate retention of the lake          --
    lakeFactorSqr         square root factor for the Modified Puls approach to calculate retention of the   --
    lakeInflowOldC        inflow to the lake from previous days                                             m/3
    lakeVolumeM3C         compressed map of lake volume                                                     m3
    lakeStorageC                                                                                            m3
    lakeOutflowC          compressed map of lake outflow                                                    m3/s
    lakeLevelC            compressed map of lake level                                                      m
    conLimitC
    normLimitC
    floodLimitC
    adjust_Normal_FloodC
    norm_floodLimitC
    minQC
    normQC
    nondmgQC
    deltaO
    deltaLN
    deltaLF
    deltaNFL
    reservoirFillC
    reservoirStorageM3C
    lakeResStorageC
    lakeResStorage
    resStorage
    waterBodyTypCTemp
    sumEvapWaterBodyC
    sumlakeResInflow
    sumlakeResOutflow
    lakeIn
    lakeEvapWaterBodyC
    resEvapWaterBodyC
    downstruct
    DtSec                 number of seconds per timestep (default = 86400)                                  s
    MtoM3                 Coefficient to change units                                                       --
    InvDtSec
    cellArea              Cell area [m²] of each simulated mesh
    UpArea1               upstream area of a grid cell                                                      m2
    lddCompress           compressed river network (without missing values)                                 --
    lakeEvaFactor         a factor which increases evaporation from lake because of wind                    --
    dtRouting             number of seconds per routing timestep                                            s
    evapWaterBodyC
    sumLakeEvapWaterBody
    noRoutingSteps
    sumResEvapWaterBodyC
    discharge             discharge                                                                         m3/s
    prelakeResStorage
    runoff
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.grid
        self.model = model

    def initWaterbodies(self):
        """
        Initialize water bodies
        Read parameters from maps e.g
        area, location, initial average discharge, type 9reservoir or lake) etc.

        Compress numpy array from mask map to the size of lakes+reservoirs
        (marked as capital C at the end of the variable name)
        """

        def buffer_waterbody(rec):
            """
            Puts a buffer of a rectangular rec around the lakes and reservoirs
            parameter rec = size of rectangular
            output buffer = compressed buffer
            """

            waterBody = decompress(self.var.waterBodyID)
            rows, cols = waterBody.shape
            buffer = np.full((rows, cols), 1.0e15)
            for y in range(rows):
                for x in range(cols):
                    id = waterBody[y, x]
                    if id > 0:
                        for j in range(1, rec + 1):
                            addj = j // 2
                            if j % 2:
                                addj = -addj
                            for i in range(1, rec + 1):
                                addi = i // 2
                                if i % 2:
                                    addi = -addi
                                yy = y + addj
                                xx = x + addi
                                if yy >= 0 and yy < rows and xx >= 0 and xx < cols:
                                    if id < buffer[yy, xx]:
                                        buffer[yy, xx] = id
            buffer[buffer == 1.0e15] = 0.0
            return compressArray(buffer).astype(np.int64)

        water_body_data = pd.read_csv(
            self.model.model_structure["table"][
                "routing/lakesreservoirs/basin_lakes_data"
            ],
            dtype={
                "waterbody_type": np.int32,
                "volume_total": np.float64,
                "average_discharge": np.float64,
                "average_area": np.float64,
                "volume_flood": np.float64,
                "relative_area_in_region": np.float64,
            },
        ).set_index("waterbody_id")

        # load lakes/reservoirs map with a single ID for each lake/reservoir
        self.var.waterBodyID = loadmap("waterBodyID").astype(np.int64)

        assert (self.var.waterBodyID >= 0).all()

        # calculate biggest outlet = biggest accumulation of ldd network
        lakeResmax = npareamaximum(self.var.UpArea1, self.var.waterBodyID)
        self.var.waterBodyOut = np.where(
            self.var.UpArea1 == lakeResmax, self.var.waterBodyID, 0
        )

        # dismiss water bodies that are not a subcatchment of an outlet
        sub = subcatchment1(self.var.dirUp, self.var.waterBodyOut, self.var.UpArea1)
        self.var.waterBodyID = np.where(self.var.waterBodyID == sub, sub, 0)

        # Create a buffer around water bodies as command areas for lakes and reservoirs
        rectangular = 1
        if "buffer_waterbodies" in binding:
            rectangular = int(loadmap("buffer_waterbodies"))
        self.var.waterBodyBuffer = buffer_waterbody(rectangular)

        # and again calculate outlets, because ID might have changed due to the operation before
        lakeResmax = npareamaximum(self.var.UpArea1, self.var.waterBodyID)
        self.var.waterBodyOut = np.where(
            self.var.UpArea1 == lakeResmax, self.var.waterBodyID, 0
        )

        # change ldd: put pits in where lakes are:
        self.var.ldd_LR = np.where(self.var.waterBodyID > 0, 5, self.var.lddCompress)

        # create new ldd without lakes reservoirs
        (
            self.var.lddCompress_LR,
            dirshort_LR,
            self.var.dirUp_LR,
            self.var.dirupLen_LR,
            self.var.dirupID_LR,
            self.var.downstruct_LR,
            self.var.catchment_LR,
            self.var.dirDown_LR,
            self.var.lendirDown_LR,
        ) = defLdd2(self.var.ldd_LR)

        # boolean map as mask map for compressing and decompressing
        self.var.compress_LR = self.var.waterBodyOut > 0
        self.var.waterBodyIDC = np.compress(self.var.compress_LR, self.var.waterBodyOut)
        self.var.decompress_LR = np.nonzero(self.var.waterBodyOut)[0]
        self.var.waterBodyOutC = np.compress(
            self.var.compress_LR, self.var.waterBodyOut
        )

        # water body types:
        # - 3 = reservoirs and lakes (used as reservoirs but before the year of construction as lakes
        # - 2 = reservoirs (regulated discharge)
        # - 1 = lakes (weirFormula)
        # - 0 = non lakes or reservoirs (e.g. wetland)
        # self.var.waterBodyTyp = loadmap('waterBodyTyp').astype(np.int64)

        # waterBodyTyp = np.where(waterBodyTyp > 0., 1, waterBodyTyp)  # TODO change all to lakes for testing
        self.var.waterBodyTypC = water_body_data.loc[self.var.waterBodyIDC][
            "waterbody_type"
        ].values
        self.var.waterBodyTypC = np.where(
            self.var.waterBodyOutC > 0, self.var.waterBodyTypC.astype(np.int64), 0
        )

        self.reservoir_operators = self.model.agents.reservoir_operators

        # ================================
        # Lakes

        # Surface area of each lake [m2]
        self.var.lakeAreaC = water_body_data.loc[self.var.waterBodyIDC][
            "average_area"
        ].values

        # FracWaterC = np.compress(self.var.compress_LR,laketotal(self.var.fracVegCover[5] * self.var.cellArea, self.var.waterBodyID))
        # if water body surface from fraction map > area from lakeres map then use fracwater map
        # not used, bc lakes res is splitted into big lakes linked to river network and small lakes linked to runoff of a gridcell

        # lake discharge at outlet to calculate alpha: parameter of channel width, gravity and weir coefficient
        # Lake parameter A (suggested  value equal to outflow width in [m])
        # multiplied with the calibration parameter LakeMultiplier
        self.var.lakeDis0C = np.maximum(
            water_body_data.loc[self.var.waterBodyIDC]["average_discharge"].values,
            0.1,
        )
        chanwidth = 7.1 * np.power(self.var.lakeDis0C, 0.539)

        self.var.lakeAC = (
            loadmap("lakeAFactor") * 0.612 * 2 / 3 * chanwidth * (2 * 9.81) ** 0.5
        )

        # ================================
        # Reservoirs
        # REM: self.var.resVolumeC = np.compress(self.var.compress_LR, loadmap('waterBodyVolRes'))
        # if vol = 0 volu = 10 * area just to mimic all lakes are reservoirs
        # in [Million m3] -> converted to mio m3

        # correcting water body types if the volume is 0:
        # correcting reservoir volume for lakes, just to run them all as reservoirs

        self.var.resVolumeC = np.zeros(self.var.waterBodyTypC.size, dtype=np.float32)
        self.var.resVolumeC[self.var.waterBodyTypC == 2] = (
            self.reservoir_operators.reservoir_volume
        )
        self.var.waterBodyTypC = np.where(
            self.var.resVolumeC > 0.0,
            self.var.waterBodyTypC,
            np.where(self.var.waterBodyTypC == 2, 1, self.var.waterBodyTypC),
        )

        # a factor which increases evaporation from lake because of wind
        self.var.lakeEvaFactor = loadmap("lakeEvaFactor")

        # initial only the big arrays are set to 0, the  initial values are loaded inside the subrouines of lakes and reservoirs
        self.var.reslakeoutflow = self.model.data.grid.full_compressed(
            0, dtype=np.float32
        )
        self.var.lakeVolume = self.model.data.grid.full_compressed(0, dtype=np.float32)
        self.var.outLake = self.var.load_initial("outLake")

        self.var.lakeStorage = self.model.data.grid.full_compressed(0, dtype=np.float32)
        self.var.lakeInflow = self.model.data.grid.full_compressed(0, dtype=np.float32)
        self.var.lakeOutflow = self.model.data.grid.full_compressed(0, dtype=np.float32)
        self.var.reservoirStorage = self.model.data.grid.full_compressed(
            0, dtype=np.float32
        )

        self.var.MtoM3C = np.compress(self.var.compress_LR, self.var.cellArea)

        # init water balance [m]
        self.var.EvapWaterBodyM = self.model.data.grid.full_compressed(
            0, dtype=np.float32
        )
        self.var.lakeResInflowM = self.model.data.grid.full_compressed(
            0, dtype=np.float32
        )
        self.var.lakeResOutflowM = self.model.data.grid.full_compressed(
            0, dtype=np.float32
        )

        if checkOption("calcWaterBalance"):
            self.var.lakedaycorrect = self.model.data.grid.full_compressed(
                0, dtype=np.float32
            )

    def initial_lakes(self):
        """
        Initial part of the lakes module
        Using the **Modified Puls approach** to calculate retention of a lake
        """

        # self_.var.lakeInflowOldC = np.bincount(self_.var.downstruct, weights=self.var.ChanQ)[self.var.LakeIndex]

        # self.var.lakeInflowOldC = np.compress(self_.var.compress_LR, self_.var.chanQKin)
        # for Modified Puls Method the Q(inflow)1 has to be used. It is assumed that this is the same as Q(inflow)2 for the first timestep
        # has to be checked if this works in forecasting mode!

        self.var.lakeFactor = self.var.lakeAreaC / (
            self.var.dtRouting * np.sqrt(self.var.lakeAC)
        )

        self.var.lakeFactorSqr = np.square(self.var.lakeFactor)
        # for faster calculation inside dynamic section

        lakeInflowIni = self.var.load_initial("lakeInflow")  # inflow in m3/s estimate
        if not (isinstance(lakeInflowIni, np.ndarray)):
            self.var.lakeInflowOldC = self.var.lakeDis0C.copy()
        else:
            self.var.lakeInflowOldC = np.compress(self.var.compress_LR, lakeInflowIni)

        lakeVolumeIni = self.var.load_initial("lakeStorage")
        if not (isinstance(lakeVolumeIni, np.ndarray)):
            self.var.lakeVolumeM3C = self.var.lakeAreaC * np.sqrt(
                self.var.lakeInflowOldC / self.var.lakeAC
            )
        else:
            self.var.lakeVolumeM3C = np.compress(self.var.compress_LR, lakeVolumeIni)
        self.var.lakeStorageC = self.var.lakeVolumeM3C.copy()

        lakeOutflowIni = self.var.load_initial("lakeOutflow")
        if not (isinstance(lakeOutflowIni, np.ndarray)):
            lakeStorageIndicator = np.maximum(
                0.0,
                self.var.lakeVolumeM3C / self.var.dtRouting
                + self.var.lakeInflowOldC / 2,
            )
            # SI = S/dt + Q/2
            self.var.lakeOutflowC = np.square(
                -self.var.lakeFactor
                + np.sqrt(self.var.lakeFactorSqr + 2 * lakeStorageIndicator)
            )
            # solution of quadratic equation
            #  it is as easy as this because:
            # 1. storage volume is increase proportional to elevation
            #  2. Q= a *H **2.0  (if you choose Q= a *H **1.5 you have to solve the formula of Cardano)
        else:
            self.var.lakeOutflowC = np.compress(self.var.compress_LR, lakeOutflowIni)

        # lake storage ini
        self.var.lakeLevelC = self.var.lakeVolumeM3C / self.var.lakeAreaC

    def initial_reservoirs(self):
        """
        Initial part of the reservoir module
        Using the appraoch of LISFLOOD

        See Also:
            LISFLOOD manual Annex 1: (Burek et al. 2013)
        """

        self.var.resVolumeC = np.zeros(self.var.waterBodyTypC.size, dtype=np.float32)
        self.var.resVolumeC[self.var.waterBodyTypC == 2] = (
            self.reservoir_operators.reservoir_volume
        )

        reservoirStorageIni = self.var.load_initial("reservoirStorage")
        if not (isinstance(reservoirStorageIni, np.ndarray)):
            self.var.reservoirFillC = np.zeros(
                self.var.waterBodyTypC.size, dtype=np.float32
            )
            self.var.reservoirFillC[self.var.waterBodyTypC == 2] = (
                self.reservoir_operators.norm_limit_ratio
            )
            # Initial reservoir fill (fraction of total storage, [-])
            self.var.reservoirStorageM3C = self.var.reservoirFillC * self.var.resVolumeC
        else:
            self.var.reservoirStorageM3C = np.compress(
                self.var.compress_LR, reservoirStorageIni
            )
            self.var.reservoirFillC = self.var.reservoirStorageM3C / self.var.resVolumeC

        # water balance
        self.var.lakeResStorageC = np.where(
            self.var.waterBodyTypC == 0,
            0.0,
            np.where(
                self.var.waterBodyTypC == 1,
                self.var.lakeStorageC,
                self.var.reservoirStorageM3C,
            ),
        )
        self.var.lakeStorageC = np.where(
            self.var.waterBodyTypC == 1, self.var.lakeStorageC, 0.0
        )
        self.var.resStorageC = np.where(
            self.var.waterBodyTypC > 1, self.var.reservoirStorageM3C, 0.0
        )
        self.var.lakeResStorage = self.model.data.grid.full_compressed(
            0, dtype=np.float32
        )
        self.var.lakeStorage = self.model.data.grid.full_compressed(0, dtype=np.float32)
        self.var.resStorage = self.model.data.grid.full_compressed(0, dtype=np.float32)
        np.put(
            self.var.lakeResStorage, self.var.decompress_LR, self.var.lakeResStorageC
        )
        np.put(self.var.lakeStorage, self.var.decompress_LR, self.var.lakeStorageC)
        np.put(self.var.resStorage, self.var.decompress_LR, self.var.resStorageC)

    def dynamic(self):
        """
        Dynamic part set lakes and reservoirs for each year
        """
        # if first timestep, or beginning of new year
        if self.model.current_timestep == 1 or (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            year = self.model.current_time.year

            # - 3 = reservoirs and lakes (used as reservoirs but before the year of construction as lakes
            # - 2 = reservoirs (regulated discharge)
            # - 1 = lakes (weirFormula)
            # - 0 = non lakes or reservoirs (e.g. wetland)
            if returnBool("DynamicResAndLakes"):
                raise NotImplementedError("DynamicResAndLakes not implemented yet")
                if returnBool("dynamicLakesRes"):
                    year = self.model.current_time.year
                else:
                    year = loadmap("fixLakesResYear")

                self.var.waterBodyTypCTemp = np.where(
                    (self.var.resYearC > year) & (self.var.waterBodyTypC == 2),
                    0,
                    self.var.waterBodyTypC,
                )
                self.var.waterBodyTypCTemp = np.where(
                    (self.var.resYearC > year) & (self.var.waterBodyTypC == 3),
                    1,
                    self.var.waterBodyTypCTemp,
                )
                # self.var.waterBodyTypTemp = np.where((self.var.resYear > year) & (self.var.waterBodyTyp == 2), 0, self.var.waterBodyTyp)
                # self.var.waterBodyTypTemp = np.where((self.var.resYear > year) & (self.var.waterBodyTyp == 3), 1, self.var.waterBodyTypTemp)
            else:
                self.var.waterBodyTypCTemp = self.var.waterBodyTypC.copy()

        self.var.sumEvapWaterBodyC = 0
        self.var.sumlakeResInflow = 0
        self.var.sumlakeResOutflow = 0

    def dynamic_inloop(self, NoRoutingExecuted):
        """
        Dynamic part to calculate outflow from lakes and reservoirs
        * lakes with modified Puls approach
        * reservoirs with special filling levels
        :param NoRoutingExecuted: actual number of routing substep
        :return: outLdd: outflow in m3 to the network
        Note:
            outflow to adjected lakes and reservoirs is calculated separately
        """

        def dynamic_inloop_lakes(inflowC, NoRoutingExecuted):
            """
            Lake routine to calculate lake outflow
            :param inflowC: inflow to lakes and reservoirs [m3]
            :param NoRoutingExecuted: actual number of routing substep
            :return: QLakeOutM3DtC - lake outflow in [m3] per subtime step
            """

            # ************************************************************
            # ***** LAKE
            # ************************************************************

            if checkOption("calcWaterBalance"):
                oldlake = self.var.lakeStorageC.copy()

            # Lake inflow in [m3/s]
            lakeInflowC = inflowC / self.var.dtRouting

            # just for day to day waterbalance -> get X as difference
            # lakeIn = in + X ->  (in + old) * 0.5 = in + X  ->   in + old = 2in + 2X -> in - 2in +old = 2x
            # -> (old - in) * 0.5 = X
            lakedaycorrectC = (
                0.5
                * (inflowC / self.var.dtRouting - self.var.lakeInflowOldC)
                * self.var.dtRouting
            )  # [m3]

            self.var.lakeIn = (lakeInflowC + self.var.lakeInflowOldC) * 0.5
            # for Modified Puls Method: (S2/dtime + Qout2/2) = (S1/dtime + Qout1/2) - Qout1 + (Qin1 + Qin2)/2
            #  here: (Qin1 + Qin2)/2

            self.var.lakeEvapWaterBodyC = np.where(
                (self.var.lakeVolumeM3C - self.var.evapWaterBodyC) > 0.0,
                self.var.evapWaterBodyC,
                self.var.lakeVolumeM3C,
            )
            self.var.sumLakeEvapWaterBodyC += self.var.lakeEvapWaterBodyC
            self.var.lakeVolumeM3C = (
                self.var.lakeVolumeM3C - self.var.lakeEvapWaterBodyC
            )
            # lakestorage - evaporation from lakes

            self.var.lakeInflowOldC = lakeInflowC.copy()
            # Qin2 becomes Qin1 for the next time step [m3/s]

            lakeStorageIndicator = np.maximum(
                0.0,
                self.var.lakeVolumeM3C / self.var.dtRouting
                - 0.5 * self.var.lakeOutflowC
                + self.var.lakeIn,
            )
            # here S1/dtime - Qout1/2 + LakeIn , so that is the right part of the equation above

            self.var.lakeOutflowC = np.square(
                -self.var.lakeFactor
                + np.sqrt(self.var.lakeFactorSqr + 2 * lakeStorageIndicator)
            )

            QLakeOutM3DtC = self.var.lakeOutflowC * self.var.dtRouting
            # Outflow in [m3] per timestep

            # New lake storage [m3] (assuming lake surface area equals bottom area)
            self.var.lakeVolumeM3C = (
                lakeStorageIndicator - self.var.lakeOutflowC * 0.5
            ) * self.var.dtRouting
            # Lake storage

            self.var.lakeStorageC += (
                self.var.lakeIn * self.var.dtRouting
                - QLakeOutM3DtC
                - self.var.lakeEvapWaterBodyC
            )

            if self.var.noRoutingSteps == (NoRoutingExecuted + 1):
                self.var.lakeLevelC = self.var.lakeVolumeM3C / self.var.lakeAreaC

            # expanding the size
            # self.var.QLakeOutM3Dt = self.model.data.grid.full_compressed(0, dtype=np.float32)
            # np.put(self.var.QLakeOutM3Dt,self.var.LakeIndex,QLakeOutM3DtC)
            # if  (self.var.noRoutingSteps == (NoRoutingExecuted + 1)):
            if self.model.save_initial_data and (
                self.var.noRoutingSteps == (NoRoutingExecuted + 1)
            ):
                np.put(
                    self.var.lakeVolume, self.var.decompress_LR, self.var.lakeVolumeM3C
                )
                np.put(
                    self.var.lakeInflow, self.var.decompress_LR, self.var.lakeInflowOldC
                )
                np.put(
                    self.var.lakeOutflow, self.var.decompress_LR, self.var.lakeOutflowC
                )

            # Water balance
            if self.var.noRoutingSteps == (NoRoutingExecuted + 1):
                np.put(
                    self.var.lakeStorage, self.var.decompress_LR, self.var.lakeStorageC
                )

            if checkOption("calcWaterBalance"):
                np.put(self.var.lakedaycorrect, self.var.decompress_LR, lakedaycorrectC)
                self.model.waterbalance_module.waterBalanceCheck(
                    influxes=[self.var.lakeIn],  # In [m3/s]
                    outfluxes=[
                        self.var.lakeOutflowC,
                        self.var.lakeEvapWaterBodyC / self.var.dtRouting,
                    ],  # Out  self.var.evapWaterBodyC
                    prestorages=[oldlake / self.var.dtRouting],  # prev storage
                    poststorages=[self.var.lakeStorageC / self.var.dtRouting],
                    name="lake",
                    tollerance=1e-5,
                )

                self.model.waterbalance_module.waterBalanceCheck(
                    influxes=[inflowC / self.var.dtRouting],  # In [m3/s]
                    outfluxes=[
                        self.var.lakeOutflowC,
                        self.var.lakeEvapWaterBodyC / self.var.dtRouting,
                        lakedaycorrectC / self.var.dtRouting,
                    ],  # Out  self.var.evapWaterBodyC
                    prestorages=[oldlake / self.var.dtRouting],  # prev storage
                    poststorages=[self.var.lakeStorageC / self.var.dtRouting],
                    name="lake2",
                    tollerance=1e-5,
                )

                self.model.waterbalance_module.waterBalanceCheck(
                    influxes=[inflowC],  # In [m3/s]
                    outfluxes=[
                        QLakeOutM3DtC,
                        self.var.lakeEvapWaterBodyC,
                        lakedaycorrectC,
                    ],  # Out  self.var.evapWaterBodyC
                    prestorages=[oldlake],  # prev storage
                    poststorages=[self.var.lakeStorageC],
                    name="lake3",
                    tollerance=0.1,
                )

            return QLakeOutM3DtC

        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------

        def dynamic_inloop_reservoirs(inflowC, NoRoutingExecuted):
            """
            Reservoir outflow
            :param inflowC: inflow to reservoirs
            :param NoRoutingExecuted: actual number of routing substep
            :return: qResOutM3DtC - reservoir outflow in [m3] per subtime step
            """

            # ************************************************************
            # ***** Reservoirs
            # ************************************************************

            if checkOption("calcWaterBalance"):
                oldres = self.var.reservoirStorageM3C.copy()

            # QResInM3Dt = inflowC
            # Reservoir inflow in [m3] per timestep
            self.var.reservoirStorageM3C += inflowC
            # New reservoir storage [m3] = plus inflow for this sub step

            # check that reservoir storage - evaporation > 0
            self.var.resEvapWaterBodyC = np.where(
                self.var.reservoirStorageM3C - self.var.evapWaterBodyC > 0.0,
                self.var.evapWaterBodyC,
                self.var.reservoirStorageM3C,
            )
            self.var.sumResEvapWaterBodyC += self.var.resEvapWaterBodyC
            self.var.reservoirStorageM3C = (
                self.var.reservoirStorageM3C - self.var.resEvapWaterBodyC
            )

            reservoirOutflow = np.zeros(self.var.waterBodyIDC.size, dtype=np.float64)
            reservoirOutflow[self.var.waterBodyTypC == 2] = (
                self.model.agents.reservoir_operators.regulate_reservoir_outflow(
                    self.var.reservoirStorageM3C[self.var.waterBodyTypC == 2],
                    inflowC[self.var.waterBodyTypC == 2]
                    / self.var.dtRouting,  # convert per timestep to per second
                    self.var.waterBodyIDC[self.var.waterBodyTypC == 2],
                )
            )

            qResOutM3DtC = reservoirOutflow * self.var.dtRouting

            # Reservoir outflow in [m3] per sub step
            qResOutM3DtC = np.where(
                self.var.reservoirStorageM3C - qResOutM3DtC > 0.0,
                qResOutM3DtC,
                self.var.reservoirStorageM3C,
            )
            # check if storage would go < 0 if outflow is used
            qResOutM3DtC = np.maximum(
                qResOutM3DtC, self.var.reservoirStorageM3C - self.var.resVolumeC
            )
            # Check to prevent reservoir storage from exceeding total capacity

            self.var.reservoirStorageM3C -= qResOutM3DtC

            self.var.reservoirStorageM3C = np.maximum(0.0, self.var.reservoirStorageM3C)

            # New reservoir storage [m3]
            self.var.reservoirFillC = self.var.reservoirStorageM3C / self.var.resVolumeC
            # New reservoir fill

            # if  (self.var.noRoutingSteps == (NoRoutingExecuted + 1)):
            if self.var.noRoutingSteps == (NoRoutingExecuted + 1):
                np.put(
                    self.var.reservoirStorage,
                    self.var.decompress_LR,
                    self.var.reservoirStorageM3C,
                )

            if checkOption("calcWaterBalance"):
                self.model.waterbalance_module.waterBalanceCheck(
                    influxes=[inflowC / self.var.dtRouting],  # In
                    outfluxes=[
                        qResOutM3DtC / self.var.dtRouting,
                        self.var.resEvapWaterBodyC / self.var.dtRouting,
                    ],  # Out  self.var.evapWaterBodyC
                    prestorages=[oldres / self.var.dtRouting],  # prev storage
                    poststorages=[self.var.reservoirStorageM3C / self.var.dtRouting],
                    name="res1",
                    tollerance=1e-4,
                )

            return qResOutM3DtC

        # ---------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------
        # lake and reservoirs

        if checkOption("calcWaterBalance"):
            prereslake = self.var.lakeResStorageC.copy()
            prelake = self.var.lakeStorageC.copy()

        # ----------
        # inflow lakes
        # 1.  dis = upstream1(self_.var.downstruct_LR, self_.var.discharge)   # from river upstream
        # 2.  runoff = laketotal(self.var_.waterBodyID, self_.var.waterBodyID)  # from cell itself
        # 3.                  # outflow from upstream lakes

        # ----------
        # outflow lakes res -> inflow ldd_LR
        # 1. out = upstream1(self_.var.downstruct, self_.var.outflow)

        # collect discharge from above waterbodies
        dis_LR = upstream1(self.var.downstruct_LR, self.var.discharge)
        # only where lakes are and unit convered to [m]
        dis_LR = np.where(self.var.waterBodyID > 0, dis_LR, 0.0) * self.model.DtSec

        # sum up runoff and discharge on the lake
        inflow = laketotal(
            dis_LR + self.var.runoff * self.var.cellArea, self.var.waterBodyID
        )

        # only once at the outlet
        inflow = (
            np.where(self.var.waterBodyOut > 0, inflow, 0.0) / self.var.noRoutingSteps
            + self.var.outLake
        )

        if checkOption("inflow"):
            # if inflow ( from module inflow) goes to a lake this is not counted, because lakes,reservoirs are dislinked from the network
            inflow2basin = laketotal(self.var.inflowDt, self.var.waterBodyID)
            inflow2basin = np.where(self.var.waterBodyOut > 0, inflow2basin, 0.0)
            inflow = inflow + inflow2basin

        # calculate total inflow into lakes and compress it to waterbodie outflow point
        # inflow to lake is discharge from upstream network + runoff directly into lake + outflow from upstream lakes
        inflowC = np.compress(self.var.compress_LR, inflow)

        # ------------------------------------------------------------
        outflowLakesC = dynamic_inloop_lakes(inflowC, NoRoutingExecuted)
        outflowResC = dynamic_inloop_reservoirs(inflowC, NoRoutingExecuted)
        outflow0C = inflowC.copy()  # no retention
        outflowC = np.where(
            self.var.waterBodyTypCTemp == 0,
            outflow0C,
            np.where(self.var.waterBodyTypCTemp == 1, outflowLakesC, outflowResC),
        )

        # outflowC =  outflowLakesC        # only lakes
        # outflowC = outflowResC
        # outflowC = inflowC.copy() - self.var.evapWaterBodyC
        # outflowC = inflowC.copy()

        # waterbalance
        inflowCorrC = np.where(
            self.var.waterBodyTypCTemp == 1,
            self.var.lakeIn * self.var.dtRouting,
            inflowC,
        )
        # EvapWaterBodyC = np.where( self.var.waterBodyTypCTemp == 0, 0. , np.where( self.var.waterBodyTypCTemp == 1, self.var.sumLakeEvapWaterBodyC, self.var.sumResEvapWaterBodyC))
        EvapWaterBodyC = np.where(
            self.var.waterBodyTypCTemp == 0,
            0.0,
            np.where(
                self.var.waterBodyTypCTemp == 1,
                self.var.lakeEvapWaterBodyC,
                self.var.resEvapWaterBodyC,
            ),
        )

        self.var.lakeResStorageC = np.where(
            self.var.waterBodyTypCTemp == 0,
            0.0,
            np.where(
                self.var.waterBodyTypCTemp == 1,
                self.var.lakeStorageC,
                self.var.reservoirStorageM3C,
            ),
        )
        self.var.lakeStorageC = np.where(
            self.var.waterBodyTypCTemp == 1, self.var.lakeStorageC, 0.0
        )
        self.var.resStorageC = np.where(
            self.var.waterBodyTypCTemp > 1, self.var.reservoirStorageM3C, 0.0
        )

        self.var.sumEvapWaterBodyC += EvapWaterBodyC  # in [m3]
        self.var.sumlakeResInflow += inflowCorrC
        self.var.sumlakeResOutflow += outflowC
        # self.var.sumlakeResOutflow = self.var.sumlakeResOutflow  + self.var.lakeOutflowC * self.var.dtRouting

        # decompress to normal maskarea size waterbalance
        if self.var.noRoutingSteps == (NoRoutingExecuted + 1):
            np.put(
                self.var.EvapWaterBodyM,
                self.var.decompress_LR,
                self.var.sumEvapWaterBodyC,
            )
            self.var.EvapWaterBodyM = self.var.EvapWaterBodyM / self.var.cellArea
            np.put(
                self.var.lakeResInflowM,
                self.var.decompress_LR,
                self.var.sumlakeResInflow,
            )
            self.var.lakeResInflowM = self.var.lakeResInflowM / self.var.cellArea
            np.put(
                self.var.lakeResOutflowM,
                self.var.decompress_LR,
                self.var.sumlakeResOutflow,
            )
            self.var.lakeResOutflowM = self.var.lakeResOutflowM / self.var.cellArea

            np.put(
                self.var.lakeResStorage,
                self.var.decompress_LR,
                self.var.lakeResStorageC,
            )
            np.put(self.var.lakeStorage, self.var.decompress_LR, self.var.lakeStorageC)
            np.put(self.var.resStorage, self.var.decompress_LR, self.var.resStorageC)

        # ------------------------------------------------------------

        np.put(self.var.reslakeoutflow, self.var.decompress_LR, outflowC)
        lakeResOutflowDis = laketotal(self.var.reslakeoutflow, self.var.waterBodyID) / (
            self.model.DtSec / self.var.noRoutingSteps
        )
        # shift outflow 1 cell downstream
        out1 = upstream1(self.var.downstruct, self.var.reslakeoutflow)
        # everything with is not going to another lake is output to river network
        outLdd = np.where(self.var.waterBodyID > 0, 0, out1)

        # everything what is not going to the network is going to another lake
        outLake1 = np.where(self.var.waterBodyID > 0, out1, 0)
        # sum up all inflow from other lakes
        outLakein = laketotal(outLake1, self.var.waterBodyID)
        # use only the value of the outflow point
        self.var.outLake = np.where(self.var.waterBodyOut > 0, outLakein, 0.0)
        if self.var.noRoutingSteps == (NoRoutingExecuted + 1) and checkOption(
            "calcWaterBalance"
        ):
            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[inflowCorrC],  # In
                outfluxes=[outflowC, EvapWaterBodyC],  # Out  EvapWaterBodyC
                prestorages=[prereslake],  # prev storage
                poststorages=[self.var.lakeResStorageC],
                tollerance=1e-5,
            )
            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[self.var.sumlakeResInflow],  # In
                outfluxes=[
                    self.var.sumlakeResOutflow,
                    self.var.sumEvapWaterBodyC,
                ],  # Out  self.var.evapWaterBodyC
                prestorages=[
                    np.compress(self.var.compress_LR, self.var.prelakeResStorage)
                ],  # prev storage
                poststorages=[self.var.lakeResStorageC],
                tollerance=1000,
            )
            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[self.var.lakeResInflowM],  # In
                outfluxes=[
                    self.var.lakeResOutflowM,
                    self.var.EvapWaterBodyM,
                ],  # Out  self.var.evapWaterBodyC
                prestorages=[
                    self.var.prelakeResStorage / self.var.cellArea
                ],  # prev storage
                poststorages=[self.var.lakeResStorage / self.var.cellArea],
                tollerance=1e-3,
            )

        return outLdd, lakeResOutflowDis
