import pandas as pd
import numpy as np
from scipy.optimize import fsolve

from .routing_reservoirs.routing_sub import (
    subcatchment1,
    define_river_network,
    upstream1,
)


def laketotal(values, areaclass, nan_class):
    """
    numpy area total procedure

    :param values:
    :param areaclass:
    :return: calculates the total area of a class

    TODO: This function can be optimized looking at the control flow
    """
    mask = areaclass != nan_class
    # add +2 to make sure that the last entry is also 0, so that 0 maps to 0
    class_totals = np.bincount(
        areaclass[mask], weights=values[mask], minlength=areaclass[mask].max() + 2
    )
    return np.take(class_totals, areaclass)


def npareamaximum(values, areaclass, na_class):
    """
    numpy area maximum procedure

    :param values:
    :param areaclass:
    :return: calculates the maximum of an area of a class
    """
    valueMax = np.zeros_like(values, shape=areaclass.max() + 1)
    if na_class:
        np.maximum.at(
            valueMax, areaclass[areaclass != na_class], values[areaclass != na_class]
        )
    else:
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

        # load lakes/reservoirs map with a single ID for each lake/reservoir
        waterBodyID = self.var.load(
            self.model.model_structure["grid"]["routing/lakesreservoirs/lakesResID"]
        )
        waterBodyID[waterBodyID == 0] = -1

        waterbody_outflow_points = self.get_outflows(waterBodyID)

        # dismiss water bodies that are not a subcatchment of an outlet
        # after this, this is the final set of water bodies
        sub = subcatchment1(
            self.var.dirUp,
            waterbody_outflow_points,
            self.var.upstream_area_n_cells,
        )
        self.var.waterBodyID_original = np.where(waterBodyID == sub, sub, -1)

        # we need to re-calculate the outflows, because the ID might have changed due
        # to the earlier operations
        waterbody_outflow_points = self.get_outflows(waterBodyID)

        compressed_waterbody_ids = np.compress(
            waterbody_outflow_points != -1, waterbody_outflow_points
        )

        self.var.waterBodyID, self.waterbody_mapping = self.map_water_bodies_IDs(
            compressed_waterbody_ids, self.var.waterBodyID_original
        )
        self.water_body_data = self.load_water_body_data(
            self.waterbody_mapping, self.var.waterBodyID_original
        )

        # we need to re-calculate the outflows, because the ID might have changed due
        # to the earlier operations. This is the final one as IDs have now been mapped
        self.var.waterbody_outflow_points = self.get_outflows(self.var.waterBodyID)

        # change ldd: put pits in where lakes are:
        ldd_LR = self.model.data.grid.decompress(
            np.where(self.var.waterBodyID != -1, 5, self.var.lddCompress), fillvalue=0
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
        self.var.compress_LR = self.var.waterbody_outflow_points != -1
        self.var.waterBodyIDC = np.compress(
            self.var.compress_LR, self.var.waterbody_outflow_points
        )

        self.var.waterBodyTypC = self.water_body_data["waterbody_type"].values

        self.reservoir_operators = self.model.agents.reservoir_operators
        self.reservoir_operators.set_reservoir_data(self.water_body_data)

        self.var.lake_area = self.water_body_data["average_area"].values
        # a factor which increases evaporation from lake because of wind TODO: use wind to set this factor
        self.var.lakeEvaFactor = self.model.config["parameters"]["lakeEvaFactor"]

        self.var.volume = self.water_body_data["volume_total"].values

        # TODO: load initial values from spinup
        self.var.storage = self.var.volume.copy() / 10

        self.var.total_inflow_from_other_water_bodies = self.var.load_initial(
            "total_inflow_from_other_water_bodies",
            default=self.var.full_compressed(0, dtype=np.float32),
        )

        # lake discharge at outlet to calculate alpha: parameter of channel width, gravity and weir coefficient
        # Lake parameter A (suggested  value equal to outflow width in [m])
        self.var.lakeDis0C = np.maximum(
            self.water_body_data["average_discharge"].values,
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
            "prev_lake_inflow", default=self.var.lakeDis0C.copy()
        )
        self.var.prev_lake_outflow = self.var.load_initial(
            "prev_lake_outflow", default=self.var.prev_lake_inflow.copy()
        )

    def map_water_bodies_IDs(self, compressed_waterbody_ids, waterBodyID_original):
        water_body_mapping = np.full(
            compressed_waterbody_ids.max() + 2, -1, dtype=np.int32
        )  # make sure that the last entry is also -1, so that -1 maps to -1
        water_body_mapping[compressed_waterbody_ids] = np.arange(
            0, compressed_waterbody_ids.size, dtype=np.int32
        )
        return water_body_mapping[waterBodyID_original], water_body_mapping

    def load_water_body_data(self, waterbody_mapping, waterbody_original_ids):
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
        )
        # drop all data that is not in the original ids
        waterbody_original_ids_compressed = np.unique(waterbody_original_ids)
        waterbody_original_ids_compressed = waterbody_original_ids_compressed[
            waterbody_original_ids_compressed != -1
        ]
        water_body_data = water_body_data[
            water_body_data["waterbody_id"].isin(waterbody_original_ids_compressed)
        ]
        # map the waterbody ids to the new ids
        water_body_data["waterbody_id"] = waterbody_mapping[
            water_body_data["waterbody_id"]
        ]
        water_body_data = water_body_data.set_index("waterbody_id")
        # sort index to align with waterBodyID
        water_body_data = water_body_data.sort_index()
        return water_body_data

    def get_outflows(self, waterBodyID):
        # calculate biggest outlet = biggest accumulation of ldd network
        lakeResmax = npareamaximum(
            self.var.upstream_area_n_cells, waterBodyID, na_class=-1
        )
        lakeResmax[waterBodyID == -1] = -1
        waterbody_outflow_points = np.where(
            self.var.upstream_area_n_cells == lakeResmax, waterBodyID, -1
        )
        # make sure that each water body has an outflow
        assert np.array_equal(
            np.unique(waterbody_outflow_points), np.unique(waterBodyID)
        )
        return waterbody_outflow_points

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
                    self.var.lake_area[lakes],
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
                influxes=[inflow_lakes],  # In [m3]
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
        if self.model.CHECK_WATER_BALANCE:
            prestorage = self.var.storage.copy()

        reservoirs = self.var.waterBodyTypC == 2

        # Reservoir inflow in [m3] per timestep
        self.var.storage[reservoirs] += inflowC[reservoirs]
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

        inflow_reservoirs = np.zeros_like(inflowC)
        inflow_reservoirs[reservoirs] = inflowC[reservoirs]
        if self.model.CHECK_WATER_BALANCE:
            self.model.waterbalance_module.waterBalanceCheck(
                influxes=[inflow_reservoirs],  # In [m3/s]
                outfluxes=[outflow_m3],
                prestorages=[prestorage],
                poststorages=[self.var.storage],
                name="reservoirs",
                tollerance=1e-7,
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
        dis_LR = np.where(self.var.waterBodyID != -1, dis_LR, 0.0) * self.model.DtSec

        # TODO: this control flow can be simplified
        # sum up runoff and discharge on the lake
        inflow = laketotal(
            dis_LR + self.var.runoff * self.var.cellArea,
            self.var.waterBodyID,
            nan_class=-1,
        )

        # only once at the outlet
        inflow = (
            np.where(self.var.waterbody_outflow_points != -1, inflow, 0.0)
            / self.var.noRoutingSteps
            + self.var.total_inflow_from_other_water_bodies
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

        outflow_grid = self.var.full_compressed(0, dtype=np.float32)
        outflow_grid[self.var.compress_LR] = outflow

        # shift outflow 1 cell downstream
        outflow_shifted_downstream = upstream1(self.var.downstruct, outflow_grid)

        # everything with is not going to another lake is output to river network
        outflow_to_river_network = np.where(
            self.var.waterBodyID != -1, 0, outflow_shifted_downstream
        )
        # everything what is not going to the network is going to another lake
        # this will be added to the inflow of the other lake in the next
        # timestep
        outflow_to_another_lake = np.where(
            self.var.waterBodyID != -1, outflow_shifted_downstream, 0
        )

        # sum up all inflow from other lakes
        total_inflow_from_other_water_bodies = laketotal(
            outflow_to_another_lake, self.var.waterBodyID, nan_class=-1
        )

        # use only the value of the outflow point
        self.var.total_inflow_from_other_water_bodies = np.where(
            self.var.waterbody_outflow_points != -1,
            total_inflow_from_other_water_bodies,
            0.0,
        )

        if self.model.CHECK_WATER_BALANCE:
            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[inflowC],  # In
                outfluxes=[outflow, evaporation],  # Out  EvapWaterBodyC
                prestorages=[prestorage],  # prev storage
                poststorages=[self.var.storage, lakedaycorrect_m3],
                tollerance=1e-3,
            )

        return outflow_to_river_network, evaporation
