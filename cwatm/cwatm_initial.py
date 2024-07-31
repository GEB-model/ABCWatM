# -------------------------------------------------------------------------
# Name:       CWATM Initial
# Purpose:
#
# Author:      PB
#
# Created:     16/05/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------


from .hydrological_modules.miscInitial import miscInitial
from .hydrological_modules.evaporationPot import evaporationPot
from .hydrological_modules.snow_frost import snow_frost
from .hydrological_modules.soil import soil
from .hydrological_modules.landcoverType import landcoverType
from .hydrological_modules.sealed_water import sealed_water
from .hydrological_modules.evaporation import evaporation
from .hydrological_modules.groundwater_modflow.modflow_transient import (
    groundwater_modflow,
)
from .hydrological_modules.water_demand import water_demand
from .hydrological_modules.interception import interception
from .hydrological_modules.runoff_concentration import runoff_concentration
from .hydrological_modules.lakes_res_small import lakes_res_small
from .hydrological_modules.waterbalance import waterbalance
from .hydrological_modules.routing_reservoirs.routing_kinematic import (
    routing_kinematic,
)
from .hydrological_modules.lakes_reservoirs import lakes_reservoirs


class CWATModel_ini:
    """
    CWATN initial part
    this part is to initialize the variables.
    It will call the initial part of the hydrological modules
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
