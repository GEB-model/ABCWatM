import numpy as np
from geb.workflows import TimingModule

from .modules.miscInitial import miscInitial
from .modules.evaporationPot import evaporationPot
from .modules.snow_frost import snow_frost
from .modules.soil import soil
from .modules.landcoverType import landcoverType
from .modules.sealed_water import sealed_water
from .modules.evaporation import evaporation
from .modules.groundwater_modflow.modflow_transient import groundwater_modflow
from .modules.water_demand import water_demand
from .modules.interception import interception
from .modules.runoff_concentration import runoff_concentration
from .modules.lakes_res_small import lakes_res_small
from .modules.waterbalance import waterbalance
from .modules.routing_reservoirs.routing_kinematic import routing_kinematic
from .modules.lakes_reservoirs import lakes_reservoirs


class CWatM:
    """
    Initial and dynamic part of the CWATM model
    * initial part takes care of all the non temporal initialiation procedures
    * dynamic part loops over time
    """

    def __init__(self):
        """
        Init part of the initial part
        defines the mask map and the outlet points
        initialization of the hydrological modules
        """

        self.misc_module = miscInitial(self)
        self.waterbalance_module = waterbalance(self)
        self.evaporationPot_module = evaporationPot(self)
        self.snowfrost_module = snow_frost(self)
        self.soil_module = soil(self)
        self.landcoverType_module = landcoverType(self)
        self.evaporation_module = evaporation(self)
        self.groundwater_modflow_module = groundwater_modflow(self)
        self.waterdemand_module = water_demand(self)
        self.interception_module = interception(self)
        self.sealed_water_module = sealed_water(self)
        self.runoff_concentration_module = runoff_concentration(self)
        self.lakes_res_small_module = lakes_res_small(self)
        self.routing_kinematic_module = routing_kinematic(self)
        self.lakes_reservoirs_module = lakes_reservoirs(self)

        self.misc_module.initial()

        self.evaporationPot_module.initial()

        ElevationStD = self.data.grid.load(
            self.model_structure["grid"]["landsurface/topo/elevation_STD"]
        )
        ElevationStD = self.data.to_HRU(data=ElevationStD, fn=None)

        self.snowfrost_module.initial(ElevationStD)
        self.soil_module.initial()

        self.landcoverType_module.initial(ElevationStD)
        self.groundwater_modflow_module.initial()
        self.interception_module.initial()

        self.runoff_concentration_module.initial()
        self.lakes_res_small_module.initial()

        self.routing_kinematic_module.initial()
        self.lakes_reservoirs_module.initWaterbodies()
        self.lakes_reservoirs_module.initial_lakes()
        self.lakes_reservoirs_module.initial_reservoirs()

        self.waterdemand_module.initial()
        self.waterbalance_module.initial()

    def dynamic(self):
        """
        Dynamic part of CWATM
        calls the dynamic part of the hydrological modules
        Looping through time and space

        Note:
            if flags set the output on the screen can be changed e.g.

            * v: no output at all
            * l: time and first gauge discharge
            * t: timing of different processes at the end
        """

        timer = TimingModule("CWatM")

        self.evaporationPot_module.dynamic()
        timer.new_split("PET")

        self.lakes_reservoirs_module.dynamic()
        timer.new_split("Waterbodies")

        self.snowfrost_module.dynamic()
        timer.new_split("Snow and frost")

        (
            interflow,
            directRunoff,
            groundwater_recharge,
            groundwater_abstraction,
            channel_abstraction,
            openWaterEvap,
            returnFlow,
        ) = self.landcoverType_module.dynamic()
        timer.new_split("Landcover")

        self.groundwater_modflow_module.dynamic(
            groundwater_recharge, groundwater_abstraction
        )
        timer.new_split("GW")

        self.runoff_concentration_module.dynamic(interflow, directRunoff)
        timer.new_split("Runoff concentration")

        self.lakes_res_small_module.dynamic()
        timer.new_split("Small waterbodies")

        self.routing_kinematic_module.dynamic(
            openWaterEvap, channel_abstraction, returnFlow
        )
        timer.new_split("Routing")

        if self.timing:
            print(timer)

    @property
    def n_individuals_per_m2(self):
        n_invidiuals_per_m2_per_HRU = np.array(
            [model.n_individuals for model in self.plantFATE if model is not None]
        )
        land_use_ratios = self.data.HRU.land_use_ratio[
            self.soil_module.plantFATE_forest_RUs
        ]
        return np.array(
            (n_invidiuals_per_m2_per_HRU * land_use_ratios).sum()
            / land_use_ratios.sum()
        )

    @property
    def biomass_per_m2(self):
        biomass_per_m2_per_HRU = np.array(
            [model.biomass for model in self.plantFATE if model is not None]
        )
        land_use_ratios = self.data.HRU.land_use_ratio[
            self.soil_module.plantFATE_forest_RUs
        ]
        return np.array(
            (biomass_per_m2_per_HRU * land_use_ratios).sum() / land_use_ratios.sum()
        )
