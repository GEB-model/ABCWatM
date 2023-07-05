# -------------------------------------------------------------------------
# Name:        Waterdemand modules
# Purpose:
#
# Author:      PB, YS, MS, JdB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import numpy as np
import cftime
import calendar
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass
from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import returnBool, binding, cbinding, loadmap, checkOption, downscale_volume
from honeybees.library.mapIO import NetCDFReader

class waterdemand_livestock:
    """
    WATERDEMAND livestock

    calculating water demand -
    livestock based on precalculated maps

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    InvCellArea           Inverse of cell area of each simulated mesh                                       m-1      
    M3toM                 Coefficient to change units                                                       --       
    domesticTime                                                                                                     
    demand_unit                                                                                                      
    livestockTime                                                                                                    
    livVar                                                                                                           
    uselivestock                                                                                                     
    livestockDemand                                                                                                  
    pot_livestockConsump                                                                                             
    liv_efficiency                                                                                                   
    ====================  ================================================================================  =========

    **Functions**
    """
    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

    def initial(self):
        """
        Initial part of the water demand module - livestock

        """

        self.livestockTime = 'monthly'
        if "livestockTimeMonthly" in binding:
            if returnBool('livestockTimeMonthly'):
                self.livestockTime = 'monthly'
            else:
                self.livestockTime = 'yearly'
        else:
            self.livestockTime = 'monthly'

        if "livestockvarname" in binding:
            self.livestockVar = cbinding("livestockvarname")
        else:
            self.livestockVar = "livestockDemand"

        self.livestock_water_demand_ds = NetCDFReader(cbinding('livestockWaterDemandFile'), self.livestockVar, xmin=self.model.xmin, xmax=self.model.xmax, ymin=self.model.ymin, ymax=self.model.ymax)
        self.livestock_water_demand_ds_SSP2 = NetCDFReader(cbinding('livestockWaterDemandFile_SSP2'), self.livestockVar, xmin=self.model.xmin, xmax=self.model.xmax, ymin=self.model.ymin, ymax=self.model.ymax)

    def dynamic(self):
        """
        Dynamic part of the water demand module - livestock
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """

        days_in_month = calendar.monthrange(self.model.current_time.year, self.model.current_time.month)[1]
        # grassland/non-irrigated land that is not owned by a crop farmer
        if self.model.args.use_gpu:
            land_use_type = self.var.land_use_type.get()
        else:
            land_use_type = self.var.land_use_type
        downscale_mask = ((land_use_type != 1) | (self.var.land_owners != -1))

        if self.model.current_time.year > 2010:
            livestock_water_demand_ds = self.livestock_water_demand_ds_SSP2
        else:
            livestock_water_demand_ds = self.livestock_water_demand_ds

        # transform from mio m3 per year (or month) to m/day
        livestock_water_demand = livestock_water_demand_ds.get_data_array(self.model.current_time.replace(day=1)) * 1_000_000 / days_in_month
        livestock_water_demand = downscale_volume(
            livestock_water_demand_ds.gt,
            self.model.data.grid.gt,
            livestock_water_demand,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.var.land_use_ratio
        )
        if self.model.args.use_gpu:
            livestock_water_demand = cp.array(livestock_water_demand)
        livestock_water_demand = self.var.M3toM(livestock_water_demand)

        efficiency = 1.
        return livestock_water_demand, efficiency