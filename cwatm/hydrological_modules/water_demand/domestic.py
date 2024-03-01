# -------------------------------------------------------------------------
# Name:        Waterdemand modules
# Purpose:
#
# Author:      PB, YS, MS, JdB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules import globals
import numpy as np
import xarray as xr
import calendar

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass
import cftime
from cwatm.management_modules.data_handling import (
    returnBool,
    binding,
    cbinding,
    divideValues,
    downscale_volume,
)


class waterdemand_domestic:
    """
    WATERDEMAND domestic

    calculating water demand -
    domenstic based on precalculated maps

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    InvCellArea           Inverse of cell area of each simulated mesh                                       m-1
    M3toM                 Coefficient to change units                                                       --
    domesticTime
    domWithdrawalVar
    domConsumptionVar
    domesticDemand
    pot_domesticConsumpt
    dom_efficiency
    demand_unit
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

    def initial(self):
        pass

    def dynamic(self):
        """
        Dynamic part of the water demand module - domestic
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """
        downscale_mask = self.var.land_use_type != 4
        if self.model.use_gpu:
            downscale_mask = downscale_mask.get()
        days_in_month = calendar.monthrange(
            self.model.current_time.year, self.model.current_time.month
        )[1]
        domestic_water_demand = (
            self.model.domestic_water_demand_ds.sel(
                time=self.model.current_time, method="ffill"
            ).domestic_water_demand
            * 1_000_000
            / days_in_month
        )
        domestic_water_demand = downscale_volume(
            self.model.domestic_water_demand_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            domestic_water_demand.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.var.land_use_ratio,
        )
        if self.model.use_gpu:
            domestic_water_demand = cp.array(domestic_water_demand)
        domestic_water_demand = self.var.M3toM(domestic_water_demand)

        domestic_water_consumption = (
            self.model.domestic_water_consumption_ds.sel(
                time=self.model.current_time, method="ffill"
            ).domestic_water_consumption
            * 1_000_000
            / days_in_month
        )
        domestic_water_consumption = downscale_volume(
            self.model.domestic_water_consumption_ds.rio.transform().to_gdal(),
            self.model.data.grid.gt,
            domestic_water_consumption.values,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.var.land_use_ratio,
        )
        if self.model.use_gpu:
            domestic_water_consumption = cp.array(domestic_water_consumption)
        domestic_water_consumption = self.var.M3toM(domestic_water_consumption)

        efficiency = divideValues(domestic_water_consumption, domestic_water_demand)
        efficiency = self.model.data.to_grid(HRU_data=efficiency, fn="max")

        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        return domestic_water_demand, efficiency
