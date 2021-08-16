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
from cwatm.management_modules.data_handling import returnBool, binding, cbinding, loadmap, readnetcdf2, divideValues, downscale_volume, checkOption
from hyve.library.mapIO import NetCDFReader

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
        self.var = model.data.subvar
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

        self.industry_water_demand_ds = NetCDFReader(cbinding('industryWaterDemandFile'), self.indWithdrawalVar, self.model.bounds)
        self.industry_water_consumption_ds = NetCDFReader(cbinding('industryWaterDemandFile'), self.indConsumptionVar, self.model.bounds)

    def dynamic(self):
        if self.industryTime == 'monthly':
            timediv = globals.dateVar['daysInMonth']
        else:
            timediv = globals.dateVar['daysInYear']
        
        if self.industryTime == 'monthly':
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
        industry_water_demand = self.industry_water_demand_ds.get_data_array(date) * 1_000_000 / timediv
        industry_water_demand = downscale_volume(
            self.industry_water_demand_ds.gt,
            self.model.data.var.gt,
            industry_water_demand,
            self.model.data.var.mask,
            self.var.var_to_subvar_uncompressed,
            downscale_mask,
            self.var.land_use_ratios
        )
        if self.model.args.use_gpu:
            industry_water_demand = cp.array(industry_water_demand)
        industry_water_demand = self.var.M3toM(industry_water_demand)

        industry_water_consumption = self.industry_water_consumption_ds.get_data_array(date) * 1_000_000 / timediv
        industry_water_consumption = downscale_volume(
            self.industry_water_consumption_ds.gt,
            self.model.data.var.gt,
            industry_water_consumption,
            self.model.data.var.mask,
            self.var.var_to_subvar_uncompressed,
            downscale_mask,
            self.var.land_use_ratios
        )
        if self.model.args.use_gpu:
            industry_water_consumption = cp.array(industry_water_consumption)
        industry_water_consumption = self.var.M3toM(industry_water_consumption)

        efficiency = divideValues(industry_water_consumption, industry_water_demand)
        efficiency = self.model.data.to_var(subdata=efficiency, fn='max')
        
        assert (efficiency <= 1).all()
        assert (efficiency >= 0).all()
        return industry_water_demand, efficiency