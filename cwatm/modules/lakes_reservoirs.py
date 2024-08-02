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
from scipy.optimize import fsolve

from .routing_reservoirs.routing_sub import (
    subcatchment1,
    define_river_network,
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


GRAVITY = 9.81
SHAPE = "rectangular"
# http://rcswww.urz.tu-dresden.de/~daigner/pdf/ueberf.pdf

if SHAPE == "rectangular":
    overflow_coefficient_mu = 0.577

    def puls_equation(new_storage, dt, lake_factor, lake_area, SI):
        return (
            new_storage / dt + (lake_factor * (new_storage / lake_area) ** 1.5) / 2 - SI
        )

else:
    raise ValueError("Invalid shape")


def get_lake_outflow_and_storage(
    dt, storage, inflow, inflow_prev, outflow_prev, lake_factor, lake_area
):
    """
    Calculate outflow and storage for a lake using the Modified Puls method

    Parameters
    ----------
    dt : float
        Time step in seconds
    storage : float
        Current storage volume in m3
    inflow : float
        Inflow to the lake in m3/s
    inflow_prev : float
        Inflow to the lake in the previous time step in m3/s
    outflow_prev : float
        Outflow from the lake in the previous time step in m3/s
    lake_factor : float
        Factor for the Modified Puls approach to calculate retention of the lake

    Returns
    -------
    outflow : float
        New outflow from the lake in m3/s
    storage : float
        New storage volume in m3

    """
    SI = (storage / dt + outflow_prev / 2) - outflow_prev + (inflow_prev + inflow) / 2
    res = fsolve(
        puls_equation, storage, args=(dt, lake_factor, lake_area, SI), full_output=True
    )
    assert res[-1] == "The solution converged."
    storage = res[0]
    outflow = lake_factor * (storage / lake_area) ** 1.5
    return outflow, storage


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

        :math:`SI = {S2 \over{dtime}} + {Qout2 \over{2}} = {A*H \over{DtRouting}} + {Qout2 \over{2}} = {A \over{DtRouting*\sqrt{a}* \sqrt{Qout2}}} + {Qout2 \over{2}}`

        -> replacement: :math:`{A \over{DtSec * \sqrt{a}}} = Lakefactor, Y = \sqrt{Q}`

        :math:`Y^2 + 2 * Lakefactor *Y - 2 * SI=0`

        solution of this quadratic equation:

        :math:`Q = (-LakeFactor + \sqrt{LakeFactor^2+2*SI})^2`


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    load_initial
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
    waterBodyTypC         water body types 3 reservoirs and lakes (used as reservoirs but before the year   --
    lakeArea              area of each lake/reservoir                                                       m2
    lakeAreaC             compressed map of the area of each lake/reservoir                                 m2
    lakeDis0              compressed map average discharge at the outlet of a lake/reservoir                m3 s-1
    lakeDis0C             average discharge at the outlet of a lake/reservoir                               m3 s-1
    outLake               outflow from lakes                                                                m
    lakeFactor            factor for the Modified Puls approach to calculate retention of the lake          --
    lake_outflow          compressed map of lake outflow                                                    m3/s
    sum_evaporation
    sum_inflow
    sum_outflow
    lakeIn
    lakeEvapWaterBodyC
    resEvapWaterBodyC
    downstruct
    lddCompress           compressed river network (without missing values)                                 --
    lakeEvaFactor         a factor which increases evaporation from lake because of wind                    --
    dtRouting             number of seconds per routing timestep                                            s
    evapWaterBodyC
    sumLakeEvapWaterBody
    noRoutingSteps
    discharge             discharge                                                                         m3/s
    runoff
    ====================  ================================================================================  =========
    """

    def __init__(self, model):
        """
        Initialize water bodies

        water body types:
        2 = reservoirs (regulated discharge)
        1 = lakes (weirFormula)
        0 = non lakes or reservoirs (e.g. wetland)
        """

        self.var = model.data.grid
        self.model = model

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
        self.var.waterBodyID = self.var.load(
            self.model.model_structure["grid"]["routing/lakesreservoirs/lakesResID"]
        ).astype(
            np.int64
        )  # not sure whether np.int64 is necessary

        assert (self.var.waterBodyID >= 0).all()

        # calculate biggest outlet = biggest accumulation of ldd network
        lakeResmax = npareamaximum(self.var.UpArea1, self.var.waterBodyID)
        self.var.waterBodyOut = np.where(
            self.var.UpArea1 == lakeResmax, self.var.waterBodyID, 0
        )

        # dismiss water bodies that are not a subcatchment of an outlet
        sub = subcatchment1(self.var.dirUp, self.var.waterBodyOut, self.var.UpArea1)
        self.var.waterBodyID = np.where(self.var.waterBodyID == sub, sub, 0)

        # and again calculate outlets, because ID might have changed due to the operation before
        lakeResmax = npareamaximum(self.var.UpArea1, self.var.waterBodyID)
        self.var.waterBodyOut = np.where(
            self.var.UpArea1 == lakeResmax, self.var.waterBodyID, 0
        )

        # change ldd: put pits in where lakes are:
        ldd_LR = self.model.data.grid.decompress(
            np.where(self.var.waterBodyID > 0, 5, self.var.lddCompress), fillvalue=0
        )

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
        ) = define_river_network(
            ldd_LR,
            self.model.data.grid,
        )

        # boolean map as mask map for compressing and decompressing
        self.var.compress_LR = self.var.waterBodyOut > 0
        self.var.waterBodyIDC = np.compress(self.var.compress_LR, self.var.waterBodyOut)
        self.var.decompress_LR = np.nonzero(self.var.waterBodyOut)[0]
        self.var.waterBodyOutC = np.compress(
            self.var.compress_LR, self.var.waterBodyOut
        )

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
        self.var.lakeAreaC = water_body_data["average_area"].values

        # a factor which increases evaporation from lake because of wind TODO: use wind to set this factor
        self.var.lakeEvaFactor = self.model.config["parameters"]["lakeEvaFactor"]

        # ================================
        # Reservoirs
        # if vol = 0 volu = 10 * area just to mimic all lakes are reservoirs
        # in [Million m3] -> converted to mio m3

        # correcting water body types if the volume is 0:
        # correcting reservoir volume for lakes, just to run them all as reservoirs
        self.var.volume = water_body_data["volume_total"].values

        # TODO: load initial values from spinup
        self.var.storage = self.var.volume

        # initial only the big arrays are set to 0, the  initial values are loaded inside the subrouines of lakes and reservoirs
        self.var.outflow = self.model.data.grid.full_compressed(0, dtype=np.float32)
        self.var.outLake = self.var.load_initial("outLake")

        # for Modified Puls Method the Q(inflow)1 has to be used. It is assumed that this is the same as Q(inflow)2 for the first timestep
        # has to be checked if this works in forecasting mode!

        # lake discharge at outlet to calculate alpha: parameter of channel width, gravity and weir coefficient
        # Lake parameter A (suggested  value equal to outflow width in [m])
        # multiplied with the calibration parameter LakeMultiplier
        self.var.lakeDis0C = np.maximum(
            water_body_data.loc[self.var.waterBodyIDC]["average_discharge"].values,
            0.1,
        )

        # channel width in [m]
        channel_width = 7.1 * np.power(self.var.lakeDis0C, 0.539)

        self.lake_factor = (
            self.model.config["parameters"]["lakeAFactor"]
            * overflow_coefficient_mu
            * (2 / 3)
            * channel_width
            * (2 * GRAVITY) ** 0.5
        )

        self.var.prev_lake_inflow = self.var.load_initial(
            "lakeInflow", default=self.var.lakeDis0C.copy()
        )
        self.var.prev_lake_outflow = self.var.load_initial(
            "lakeOutflow", default=self.var.prev_lake_inflow.copy()
        )

        return None

    def step(self):
        """
        Dynamic part set lakes and reservoirs for each year
        """
        # if first timestep, or beginning of new year
        if self.model.current_timestep == 1 or (
            self.model.current_time.month == 1 and self.model.current_time.day == 1
        ):
            # - 3 = reservoirs and lakes (used as reservoirs but before the year of construction as lakes
            # - 2 = reservoirs (regulated discharge)
            # - 1 = lakes (weirFormula)
            # - 0 = non lakes or reservoirs (e.g. wetland)
            if self.model.DynamicResAndLakes:
                raise NotImplementedError("DynamicResAndLakes not implemented yet")

        self.sum_evaporation = 0
        self.sum_inflow = 0
        self.sum_outflow = 0

    def dynamic_inloop_lakes(self, inflowC):
        """
        Lake routine to calculate lake outflow
        :param inflowC: inflow to lakes and reservoirs [m3]
        :param NoRoutingExecuted: actual number of routing substep
        :return: QLakeOutM3DtC - lake outflow in [m3] per subtime step
        """
        if self.model.CHECK_WATER_BALANCE:
            prestorage = self.var.storage.copy()

        lakes = self.var.waterBodyTypC == 1

        # Lake inflow in [m3/s]
        lake_inflow_m3_s = np.zeros_like(inflowC)
        lake_inflow_m3_s[lakes] = inflowC[lakes] / self.var.dtRouting

        self.var.lake_outflow = np.zeros_like(inflowC)

        # check if there are any lakes in the model
        if (lakes == True).any():
            self.var.lake_outflow[lakes], self.var.storage[lakes] = (
                get_lake_outflow_and_storage(
                    self.var.dtRouting,
                    self.var.storage[lakes],
                    lake_inflow_m3_s[lakes],
                    self.var.prev_lake_inflow[lakes],
                    self.var.prev_lake_outflow[lakes],
                    self.lake_factor[lakes],
                    self.var.lakeAreaC[lakes],
                )
            )

        # Difference between current and previous inflow
        lakedaycorrect_m3 = np.zeros_like(inflowC)
        lakedaycorrect_m3[lakes] = (
            (lake_inflow_m3_s[lakes] - self.var.prev_lake_inflow[lakes])
            * 0.5
            * self.var.dtRouting
        ) - (
            (self.var.lake_outflow[lakes] - self.var.prev_lake_outflow[lakes])
            * 0.5
            * self.var.dtRouting
        )  # [m3]

        outflow_m3 = self.var.lake_outflow * self.var.dtRouting

        if self.model.CHECK_WATER_BALANCE:
            self.model.waterbalance_module.waterBalanceCheck(
                influxes=[
                    (lake_inflow_m3_s + self.var.prev_lake_inflow) / 2
                ],  # In [m3/s]
                outfluxes=[
                    (self.var.lake_outflow + self.var.prev_lake_outflow) / 2,
                ],  # Out
                prestorages=[prestorage / self.var.dtRouting],
                poststorages=[self.var.storage / self.var.dtRouting],
                name="lake",
                tollerance=1e-5,
            )

            inflow_lakes = np.zeros_like(inflowC)
            inflow_lakes[lakes] = inflowC[lakes]
            self.model.waterbalance_module.waterBalanceCheck(
                influxes=[inflow_lakes / self.var.dtRouting],  # In [m3/s]
                outfluxes=[
                    self.var.lake_outflow,
                    lakedaycorrect_m3 / self.var.dtRouting,
                ],
                prestorages=[prestorage / self.var.dtRouting],
                poststorages=[self.var.storage / self.var.dtRouting],
                name="lake2",
                tollerance=1e-5,
            )

            self.model.waterbalance_module.waterBalanceCheck(
                influxes=[inflow_lakes],  # In [m3/s]
                outfluxes=[
                    outflow_m3,
                    lakedaycorrect_m3,
                ],
                prestorages=[prestorage],
                poststorages=[self.var.storage],
                name="lake3",
                tollerance=0.1,
            )

        self.var.prev_lake_inflow = lake_inflow_m3_s
        self.var.prev_lake_outflow = self.var.lake_outflow

        return outflow_m3, lakedaycorrect_m3

    def dynamic_inloop_reservoirs(self, inflowC):
        """
        Reservoir outflow
        :param inflowC: inflow to reservoirs
        :return: qResOutM3DtC - reservoir outflow in [m3] per subtime step
        """

        # ************************************************************
        # ***** Reservoirs
        # ************************************************************

        if self.model.CHECK_WATER_BALANCE:
            prestorage = self.var.storage.copy()

        reservoirs = self.var.waterBodyTypC == 2

        # Reservoir inflow in [m3] per timestep
        self.var.storage += inflowC
        # New reservoir storage [m3] = plus inflow for this sub step

        outflow_m3_s = np.zeros(self.var.waterBodyIDC.size, dtype=np.float64)
        outflow_m3_s[reservoirs] = (
            self.model.agents.reservoir_operators.regulate_reservoir_outflow(
                self.var.storage[reservoirs],
                inflowC[reservoirs]
                / self.var.dtRouting,  # convert per timestep to per second
                self.var.waterBodyIDC[reservoirs],
            )
        )

        outflow_m3 = outflow_m3_s * self.var.dtRouting
        assert (outflow_m3 <= self.var.storage).all()

        self.var.storage -= outflow_m3

        if self.model.CHECK_WATER_BALANCE:
            self.model.waterbalance_module.waterBalanceCheck(
                influxes=[inflowC],  # In [m3/s]
                outfluxes=[outflow_m3],
                prestorages=[prestorage],
                poststorages=[self.var.storage],
                name="reservoirs",
                tollerance=0.1,
            )

        return outflow_m3

    def dynamic_inloop(self, step):
        """
        Dynamic part to calculate outflow from lakes and reservoirs
        * lakes with modified Puls approach
        * reservoirs with special filling levels
        :param NoRoutingExecuted: actual number of routing substep
        :return: outLdd: outflow in m3 to the network
        Note:
            outflow to adjected lakes and reservoirs is calculated separately
        """

        if self.model.CHECK_WATER_BALANCE:
            prestorage = self.var.storage.copy()

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

        # evaporation from water body
        # TODO: Abstract evaporation in lakes reservoir module for control flow simplification
        # lakeEvaFactorC
        evaporation = np.where(
            self.var.waterBodyTypC != 0,
            np.minimum(
                self.var.evapWaterBodyC, self.var.storage
            ),  # evaporation is already in m3 per routing substep
            0,
        )
        self.var.storage -= evaporation

        # calculate total inflow into lakes and compress it to waterbodie outflow point
        # inflow to lake is discharge from upstream network + runoff directly into lake + outflow from upstream lakes
        inflowC = np.compress(self.var.compress_LR, inflow)
        # ------------------------------------------------------------
        outflow_lakes, lakedaycorrect_m3 = self.dynamic_inloop_lakes(inflowC)
        outflow_reservoirs = self.dynamic_inloop_reservoirs(inflowC)

        outflow = outflow_lakes + outflow_reservoirs

        self.sum_evaporation += evaporation  # in [m3]
        self.sum_inflow += inflowC
        self.sum_outflow += outflow

        np.put(self.var.outflow, self.var.decompress_LR, outflow)
        # shift outflow 1 cell downstream
        # TODO: Check this in GUI
        self.var.outflow_shifted = upstream1(self.var.downstruct, self.var.outflow)

        # everything with is not going to another lake is output to river network
        outflow_to_river_network = np.where(
            self.var.waterBodyID > 0, 0, self.var.outflow_shifted
        )
        # everything what is not going to the network is going to another lake
        # this will be added to the inflow of the other lake in the next
        # timestep
        outflow_to_another_lake = np.where(
            self.var.waterBodyID > 0, self.var.outflow_shifted, 0
        )

        # sum up all inflow from other lakes
        outLake = laketotal(outflow_to_another_lake, self.var.waterBodyID)
        if outLake.sum() > 0:
            print("This MUST be checked before going on ...")
        # use only the value of the outflow point
        self.var.outLake = np.where(self.var.waterBodyOut > 0, outLake, 0.0)

        if self.model.CHECK_WATER_BALANCE:
            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[inflowC],  # In
                outfluxes=[outflow, evaporation],  # Out  EvapWaterBodyC
                prestorages=[prestorage],  # prev storage
                poststorages=[self.var.storage, lakedaycorrect_m3],
                tollerance=1e-5,
            )

        return outflow_to_river_network, outflow_to_river_network
