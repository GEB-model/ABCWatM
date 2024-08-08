import sys
import os

import numpy as np
from geb.workflows import TimingModule

from .modules.miscInitial import miscInitial
from .modules.evaporationPot import evaporationPot
from .modules.snow_frost import snow_frost
from .modules.soil import soil
from .modules.landcoverType import landcoverType
from .modules.sealed_water import sealed_water
from .modules.evaporation import evaporation
from .modules.groundwater import groundwater
from .modules.water_demand import water_demand
from .modules.interception import interception
from .modules.runoff_concentration import runoff_concentration
from .modules.lakes_res_small import lakes_res_small
from .modules.waterbalance import waterbalance
from .modules.routing.routing_kinematic import routing_kinematic
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
        self.init_water_table_file = os.path.join(
            self.config["general"]["init_water_table"]
        )
        self.DynamicResAndLakes = False
        self.useSmallLakes = False
        self.CHECK_WATER_BALANCE = sys.flags.optimize == 0  # 0, means no optimizations
        self.crop_factor_calibration_factor = 1
        self.soilLayers = 3

        ElevationStD = self.data.grid.load(
            self.model_structure["grid"]["landsurface/topo/elevation_STD"]
        )
        ElevationStD = self.data.to_HRU(data=ElevationStD, fn=None)

        self.misc_module = miscInitial(self)
        self.waterbalance_module = waterbalance(self)
        self.evaporationPot_module = evaporationPot(self)
        self.snowfrost_module = snow_frost(self, ElevationStD)
        self.soil_module = soil(self)
        self.landcoverType_module = landcoverType(self, ElevationStD)
        self.evaporation_module = evaporation(self)
        self.groundwater_module = groundwater(self)
        self.interception_module = interception(self)
        self.sealed_water_module = sealed_water(self)
        self.runoff_concentration_module = runoff_concentration(self)
        self.lakes_res_small_module = lakes_res_small(self)
        self.routing_kinematic_module = routing_kinematic(self)
        self.lakes_reservoirs_module = lakes_reservoirs(self)
        self.waterdemand_module = water_demand(self)

    def step(self):
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

        self.evaporationPot_module.step()
        timer.new_split("PET")

        self.lakes_reservoirs_module.step()
        timer.new_split("Waterbodies")

        self.snowfrost_module.step()
        timer.new_split("Snow and frost")

        (
            interflow,
            directRunoff,
            groundwater_recharge,
            groundwater_abstraction,
            channel_abstraction,
            openWaterEvap,
            returnFlow,
        ) = self.landcoverType_module.step()
        timer.new_split("Landcover")

        self.groundwater_module.step(groundwater_recharge, groundwater_abstraction)
        timer.new_split("GW")

        self.runoff_concentration_module.step(interflow, directRunoff)
        timer.new_split("Runoff concentration")

        self.lakes_res_small_module.step()
        timer.new_split("Small waterbodies")

        self.routing_kinematic_module.step(
            openWaterEvap, channel_abstraction, returnFlow
        )
        timer.new_split("Routing")

        if self.timing:
            print(timer)

    def finalize(self) -> None:
        """
        Finalize the model
        """
        # finalize modflow model
        self.groundwater_module.modflow.finalize()

        if self.config["general"]["simulate_forest"]:
            for plantFATE_model in self.model.plantFATE:
                if plantFATE_model is not None:
                    plantFATE_model.finalize()

    def export_water_table(self) -> None:
        """Function to save required water table output to file."""
        dirname = os.path.dirname(self.init_water_table_file)
        os.makedirs(dirname, exist_ok=True)
        np.save(
            self.init_water_table_file,
            self.groundwater_module.modflow.decompress(
                self.groundwater_module.modflow.head
            ),
        )

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
