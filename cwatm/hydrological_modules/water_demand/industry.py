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
except (ModuleNotFoundError, ImportError):
    pass
import cftime
import calendar
from cwatm.management_modules.data_handling import returnBool, binding, cbinding, loadmap, divideValues, downscale_volume, checkOption
from honeybees.library.mapIO import NetCDFReader

class waterdemand_industry:
    """
    WATERDEMAND industry

    calculating water demand -
    industry based on precalculated maps

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    InvCellArea           Inverse of cell area of each simulated mesh                                       m-1      
    M3toM                 Coefficient to change units                                                       --       
    demand_unit                                                                                                      
    industryTime                                                                                                     
    indWithdrawalVar                                                                                                 
    indConsumptionVar                                                                                                
    industryDemand                                                                                                   
    pot_industryConsumpt                                                                                             
    ind_efficiency                                                                                                   
    ====================  ================================================================================  =========

    **Functions**
    """
    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

    def initial(self):
        """
        Initial part of the water demand module - industry

        """

        if "industryTimeMonthly" in binding:
            if returnBool('industryTimeMonthly'):
                self.industryTime = 'monthly'
            else:
                self.industryTime = 'yearly'
        else:
            self.industryTime = 'monthly'

        if "industryWithdrawalvarname" in binding:
            self.indWithdrawalVar = cbinding("industryWithdrawalvarname")
        else:
            self.indWithdrawalVar = "industryGrossDemand"
        if "industryConsuptionvarname" in binding:
            self.indConsumptionVar = cbinding("industryConsuptionvarname")
        else:
            self.indConsumptionVar = "industryNettoDemand"

        self.industry_water_demand_ds = NetCDFReader(cbinding('industryWaterDemandFile'), self.indWithdrawalVar, xmin=self.model.xmin, xmax=self.model.xmax, ymin=self.model.ymin, ymax=self.model.ymax)
        self.industry_water_consumption_ds = NetCDFReader(cbinding('industryWaterDemandFile'), self.indConsumptionVar, xmin=self.model.xmin, xmax=self.model.xmax, ymin=self.model.ymin, ymax=self.model.ymax)
        self.industry_water_demand_ds_SSP2 = NetCDFReader(cbinding('industryWaterDemandFile_SSP2'), self.indWithdrawalVar, xmin=self.model.xmin, xmax=self.model.xmax, ymin=self.model.ymin, ymax=self.model.ymax)
        self.industry_water_consumption_ds_SSP2 = NetCDFReader(cbinding('industryWaterDemandFile_SSP2'), self.indConsumptionVar, xmin=self.model.xmin, xmax=self.model.xmax, ymin=self.model.ymin, ymax=self.model.ymax)

    def dynamic(self):
        downscale_mask = (self.var.land_use_type != 4)
        if self.model.args.use_gpu: 
            downscale_mask = downscale_mask.get()

        days_in_year = 366 if calendar.isleap(self.model.current_time.year) else 365
        
        # transform from mio m3 per year (or month) to m/day
        if self.model.current_time.year > 2010:
            industry_water_demand_ds = self.industry_water_demand_ds_SSP2
        else:
            industry_water_demand_ds = self.industry_water_demand_ds
        industry_water_demand = industry_water_demand_ds.get_data_array(self.model.current_time.replace(month=1, day=1)) * 1_000_000 / days_in_year
        industry_water_demand = downscale_volume(
            self.industry_water_demand_ds.gt,
            self.model.data.grid.gt,
            industry_water_demand,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.var.land_use_ratio
        )
        if self.model.args.use_gpu:
            industry_water_demand = cp.array(industry_water_demand)
        industry_water_demand = self.var.M3toM(industry_water_demand)

        if self.model.current_time.year > 2010:
            industry_water_consumption_ds = self.industry_water_consumption_ds_SSP2
        else:
            industry_water_consumption_ds = self.industry_water_consumption_ds
        industry_water_consumption = industry_water_consumption_ds.get_data_array(self.model.current_time.replace(month=1, day=1)) * 1_000_000 / days_in_year
        industry_water_consumption = downscale_volume(
            self.industry_water_consumption_ds.gt,
            self.model.data.grid.gt,
            industry_water_consumption,
            self.model.data.grid.mask,
            self.model.data.grid_to_HRU_uncompressed,
            downscale_mask,
            self.var.land_use_ratio
        )
        if self.model.args.use_gpu:
            industry_water_consumption = cp.array(industry_water_consumption)
        industry_water_consumption = self.var.M3toM(industry_water_consumption)

        efficiency = divideValues(industry_water_consumption, industry_water_demand)
        efficiency = self.model.data.to_grid(HRU_data=efficiency, fn='max')
        
        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        return industry_water_demand, efficiency