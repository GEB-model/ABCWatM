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
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
import cftime
from cwatm.management_modules.data_handling import returnBool, binding, cbinding, divideValues, downscale_volume, checkOption
from hyve.library.mapIO import NetCDFReader

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
        self.var = model.data.landunit
        self.model = model

    def initial(self):
        """
        Initial part of the water demand module

        """

        if "domesticTimeMonthly" in binding:
            if returnBool('domesticTimeMonthly'):
                self.domesticTime = 'monthly'
            else:
                self.domesticTime = 'yearly'
        else:
            self.domesticTime = 'monthly'

        if "domesticWithdrawalvarname" in binding:
            self.domWithdrawalVar = cbinding("domesticWithdrawalvarname")
        else:
            self.domWithdrawalVar = "domesticGrossDemand"
        if "domesticConsuptionvarname" in binding:
            self.domConsumptionVar = cbinding("domesticConsuptionvarname")
        else:
            self.domConsumptionVar = "domesticNettoDemand"
        
        self.domestic_water_demand_ds = NetCDFReader(cbinding('domesticWaterDemandFile'), self.domWithdrawalVar, self.model.bounds)
        self.domestic_water_consumption_ds = NetCDFReader(cbinding('domesticWaterDemandFile'), self.domConsumptionVar, self.model.bounds)

    def dynamic(self):
        """
        Dynamic part of the water demand module - domestic
        read monthly (or yearly) water demand from netcdf and transform (if necessary) to [m/day]

        """

        if self.domesticTime == 'monthly':
            timediv = globals.dateVar['daysInMonth']
        else:
            timediv = globals.dateVar['daysInYear']
        
        if self.domesticTime == 'monthly':
            date = cftime.Datetime360Day(
                globals.dateVar['currDate'].year,
                globals.dateVar['currDate'].month,
                1
            )
        else:
            date = cftime.Datetime360Day(
                globals.dateVar['currDate'].year,
                1,
                1
            )
        downscale_mask = (self.var.land_use_type != 4)
        if self.model.args.use_gpu:
            downscale_mask = downscale_mask.get()
        

        # transform from mio m3 per year (or month) to m/day
        domestic_water_demand = self.domestic_water_demand_ds.get_data_array(date) * 1_000_000 / timediv
        domestic_water_demand = downscale_volume(
            self.domestic_water_demand_ds.gt,
            self.model.data.var.gt,
            domestic_water_demand,
            self.model.data.var.mask,
            self.var.var_to_landunit_uncompressed,
            downscale_mask,
            self.var.land_use_ratio
        )
        if self.model.args.use_gpu:
            domestic_water_demand = cp.array(domestic_water_demand)
        domestic_water_demand = self.var.M3toM(domestic_water_demand)

        domestic_water_consumption = self.domestic_water_consumption_ds.get_data_array(date) * 1_000_000 / timediv
        domestic_water_consumption = downscale_volume(
            self.domestic_water_consumption_ds.gt,
            self.model.data.var.gt,
            domestic_water_consumption,
            self.model.data.var.mask,
            self.var.var_to_landunit_uncompressed,
            downscale_mask,
            self.var.land_use_ratio
        )
        if self.model.args.use_gpu:
            domestic_water_consumption = cp.array(domestic_water_consumption)
        domestic_water_consumption = self.var.M3toM(domestic_water_consumption)

        efficiency = divideValues(domestic_water_consumption, domestic_water_demand)
        efficiency = self.model.data.to_var(landunit_data=efficiency, fn='max')
        
        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        return domestic_water_demand, efficiency