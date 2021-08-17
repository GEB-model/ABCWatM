 # -------------------------------------------------------------------------
# Name:        Waterdemand module
# Purpose:
#
# Author:      PB, JB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
import rasterio
from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import option, cbinding, loadmap, checkOption
from cwatm.hydrological_modules.water_demand.domestic import waterdemand_domestic
from cwatm.hydrological_modules.water_demand.industry import waterdemand_industry
from cwatm.hydrological_modules.water_demand.livestock import waterdemand_livestock
from cwatm.hydrological_modules.water_demand.irrigation import waterdemand_irrigation
from cwatm.hydrological_modules.water_demand.environmental_need import waterdemand_environmental_need


class water_demand:
    """
    WATERDEMAND

    calculating water demand -
    Industrial, domenstic on precalculated maps
    Agricultural water demand based on water need by plants
    
    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    readAvlStorGroundwat  same as storGroundwater but equal to 0 when inferior to a treshold                m        
    nonFossilGroundwater  groundwater abstraction which is sustainable and not using fossil resources       m        
    waterbalance_module                                                                                              
    waterBodyID           lakes/reservoirs map with a single ID for each lake/reservoir                     --       
    compress_LR           boolean map as mask map for compressing lake/reservoir                            --       
    decompress_LR         boolean map as mask map for decompressing lake/reservoir                          --       
    MtoM3                 Coefficient to change units                                                       --       
    lakeVolumeM3C         compressed map of lake volume                                                     m3       
    lakeStorageC                                                                                            m3       
    reservoirStorageM3C                                                                                              
    lakeResStorage                                                                                                   
    waterBodyTypCTemp                                                                                                
    InvDtSec                                                                                                         
    cellArea              Cell area [m²] of each simulated mesh                                                      
    smalllakeVolumeM3                                                                                                
    smalllakeStorage                                                                                                 
    act_SurfaceWaterAbst                                                                                             
    fracVegCover          Fraction of area covered by the corresponding landcover type                               
    addtoevapotrans                                                                                                  
    M3toM                 Coefficient to change units                                                       --       
    act_irrConsumption    actual irrgation water consumption                                                m        
    channelStorageM3                                                                                                   
    act_bigLakeResAbst                                                                                               
    act_smallLakeResAbst                                                                                             
    returnFlow                                                                                                       
    modflowTopography                                                                                                
    modflowDepth2                                                                                                    
    leakageC                                                                                                         
    domestic_water_demand                                                                                                   
    pot_domesticConsumpt                                                                                             
    dom_efficiency                                                                                                   
    demand_unit                                                                                                      
    envFlow                                                                                                          
    industry_water_demand                                                                                                   
    pot_industryConsumpt                                                                                             
    ind_efficiency                                                                                                   
    unmetDemandPaddy                                                                                                 
    unmetDemandNonpaddy                                                                                              
    unmetDemand                                                                                                      
    efficiencyPaddy                                                                                                  
    efficiencyNonpaddy                                                                                               
    returnfractionIrr                                                                                                
    irrDemand                                                                                                        
    totalIrrDemand                                                                                                   
    livestock_water_demand                                                                                                  
    pot_livestockConsump                                                                                             
    liv_efficiency                                                                                                   
    allocSegments                                                                                                    
    swAbstractionFractio                                                                                             
    leakage                                                                                                          
    nonIrrReturnFlowFrac                                                                                             
    nonIrruse                                                                                                        
    act_indDemand                                                                                                    
    act_domDemand                                                                                                    
    act_livDemand                                                                                                    
    nonIrrDemand                                                                                                     
    totalWaterDemand                                                                                                 
    act_irrWithdrawal                                                                                                
    act_nonIrrWithdrawal                                                                                             
    act_totalWaterWithdr                                                                                             
    act_indConsumption                                                                                               
    act_domConsumption                                                                                               
    act_livConsumption                                                                                               
    act_nonIrrConsumptio                                                                                             
    act_totalIrrConsumpt                                                                                             
    act_totalWaterConsum                                                                                             
    returnflowIrr                                                                                                    
    pot_nonIrrConsumptio                                                                                             
    readAvlChannelStorag                                                                                             
    reservoir_command_ar                                                                                             
    leakageC_daily                                                                                                   
    leakageC_daily_segme                                                                                             
    pot_GroundwaterAbstr                                                                                             
    renewableAvlWater                                                                                                
    act_irrNonpaddyWithd                                                                                             
    act_irrPaddyWithdraw                                                                                             
    act_irrPaddyDemand                                                                                               
    act_irrNonpaddyDeman                                                                                             
    act_indWithdrawal                                                                                                
    act_domWithdrawal                                                                                                
    act_livWithdrawal                                                                                                
    waterDemandLost                                                                                                  
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.model = model
        self.var = model.data.landunit
        self.farmers = model.agents.farmers

        self.domestic = waterdemand_domestic(model)
        self.industry = waterdemand_industry(model)
        self.livestock = waterdemand_livestock(model)
        self.irrigation = waterdemand_irrigation(model)
        self.environmental_need = waterdemand_environmental_need(model)

    def initial(self):
        """
        Initial part of the water demand module

        Set the water allocation
        """

        if checkOption('includeWaterDemand'):
          
            self.domestic.initial()
            self.industry.initial()
            self.livestock.initial()
            self.irrigation.initial()
            self.environmental_need.initial()

            if checkOption('includeWaterBodies'): 
                self.var.using_reservoir_command_areas = False
                if 'using_reservoir_command_areas' in option:
                    if checkOption('using_reservoir_command_areas'):
                        self.var.using_reservoir_command_areas = True
                        with rasterio.open(cbinding('reservoir_command_areas'), 'r') as src:
                            reservoir_command_areas = self.model.data.var.compress(src.read(1))
                            reservoir_command_areas_mapped = self.model.data.var.water_body_mapping[reservoir_command_areas]
                            reservoir_command_areas_mapped[reservoir_command_areas == -1] = -1
                            self.var.reservoir_command_areas = self.model.data.to_landunit(data=reservoir_command_areas_mapped)

                self.var.using_lift_command_areas = False
                if 'using_lift_command_areas' in option:
                    if checkOption('using_lift_command_areas'):
                        self.var.using_lift_command_areas = True 
                        lift_command_areas = loadmap('lift_command_areas').astype(np.int)
                        self.var.lift_command_areas = self.model.data.to_landunit(data=lift_command_areas, fn=None)
            else:
                self.var.reservoir_command_areas = self.var.full_compressed(-1, dtype=np.int32)

            # self.var.crops = self.var.full_compressed(0, dtype=np.float32)
            # self.var.head2 = 0
            # self.var.demand_Segment = self.var.full_compressed(0, dtype=np.float32)
            # self.model.data.var.lakeResStorage_ratio_CA = self.model.data.var.full_compressed(0, dtype=np.float32)
            # self.model.data.var.lakeResStorage_ratio = self.model.data.var.full_compressed(0, dtype=np.float32)
            # self.model.data.var.act_bigLakeResAbst_alloc = self.model.data.var.full_compressed(0, dtype=np.float32)
            # self.var.act_channelAbstract = self.var.full_compressed(0, dtype=np.float32)
            # self.model.data.var.act_LocalLakeAbstract = self.model.data.var.full_compressed(0, dtype=np.float32)
            if checkOption('canal_leakage'):
                self.model.data.var.leakageC = np.compress(self.model.data.var.compress_LR, self.model.data.var.full_compressed(0, dtype=np.float32))
            # self.var.delivered_water = self.var.full_compressed(0, dtype=np.float32)
            # self.var.act_irrWithdrawalSW = self.var.full_compressed(0, dtype=np.float32)
            # self.var.act_irrWithdrawalGW = self.var.full_compressed(0, dtype=np.float32)
            # self.var.act_nonIrrWithdrawalSW = self.var.full_compressed(0, dtype=np.float32)
            # self.var.act_nonIrrWithdrawalGW = self.var.full_compressed(0, dtype=np.float32)
            # self.var.availableGWStorageFraction = self.var.full_compressed(0, dtype=np.float32)

            self.model.data.var.gwstorage_full = float(cbinding('poro'))* float(cbinding('thickness'))+globals.inZero

    def get_available_water(self):
        def get_available_water_reservoir_command_areas():
            # day_of_year = globals.dateVar['currDate'].timetuple().tm_yday  # Jan 1 is 1
            return self.model.data.var.reservoirStorageM3C * float(cbinding('max_reseroir_release_factor'))
            # return reservoir_storage_per_command_area * self.max_reservoir_releases[day_of_year - 1]  # remove 1 because Jan 1 is 1

        available_reservoir_storage_m3 = get_available_water_reservoir_command_areas()
        return self.model.data.var.channelStorageM3.copy(), available_reservoir_storage_m3, self.model.groundwater_modflow_module.available_groundwater_m, self.model.data.var.head

    def withdraw(self, source, demand):
        withdrawal = np.minimum(source, demand)
        source -= withdrawal  # update in place
        demand -= withdrawal  # update in place
        return withdrawal

    def dynamic(self, totalPotET):
        """
        Dynamic part of the water demand module

        * calculate the fraction of water from surface water vs. groundwater
        * get non-Irrigation water demand and its return flow fraction
        """

        if checkOption('includeWaterDemand'):

            # WATER DEMAND
            domestic_water_demand, domestic_water_efficiency = self.domestic.dynamic()
            industry_water_demand, industry_water_efficiency = self.industry.dynamic()
            livestock_water_demand, livestock_water_efficiency = self.livestock.dynamic()
            pot_irrConsumption = self.irrigation.dynamic(totalPotET)

            assert (domestic_water_demand >= 0).all()
            assert (industry_water_demand >= 0).all()
            assert (livestock_water_demand >= 0).all()
            assert (pot_irrConsumption >= 0).all()

            available_channel_storage_m3, available_reservoir_storage_m3, available_groundwater_m, groundwater_head = self.get_available_water()
            available_groundwater_m3 = self.model.data.var.MtoM3(available_groundwater_m)
            
            available_channel_storage_m3_pre = available_channel_storage_m3.copy()
            available_reservoir_storage_m3_pre = available_reservoir_storage_m3.copy()
            available_groundwater_m3_pre = available_groundwater_m3.copy()
             
            # water withdrawal
            # 1. domestic (surface + ground)
            domestic_water_demand = self.model.data.to_var(landunit_data=domestic_water_demand, fn='mean')
            domestic_water_demand_m3 = self.model.data.var.MtoM3(domestic_water_demand)
            del domestic_water_demand
            
            domestic_withdrawal_m3 = self.withdraw(available_channel_storage_m3, domestic_water_demand_m3)  # withdraw from surface water
            domestic_withdrawal_m3 += self.withdraw(available_groundwater_m3, domestic_water_demand_m3)  # withdraw from groundwater
            domestic_return_flow_m = self.model.data.var.M3toM(domestic_withdrawal_m3 * (1 - domestic_water_efficiency))

            # 2. industry (surface + ground)
            industry_water_demand = self.model.data.to_var(landunit_data=industry_water_demand, fn='mean')
            industry_water_demand_m3 = self.model.data.var.MtoM3(industry_water_demand)
            del industry_water_demand
            
            industry_withdrawal_m3 = self.withdraw(available_channel_storage_m3, industry_water_demand_m3)  # withdraw from surface water
            industry_withdrawal_m3 += self.withdraw(available_groundwater_m3, industry_water_demand_m3)  # withdraw from groundwater
            industry_return_flow_m = self.model.data.var.M3toM(industry_withdrawal_m3 * (1 - industry_water_efficiency))

            # 3. livestock (surface)
            livestock_water_demand = self.model.data.to_var(landunit_data=livestock_water_demand, fn='mean')
            livestock_water_demand_m3 = self.model.data.var.MtoM3(livestock_water_demand)
            del livestock_water_demand
            
            livestock_withdrawal_m3 = self.withdraw(available_channel_storage_m3, livestock_water_demand_m3)  # withdraw from surface water
            livestock_return_flow_m = self.model.data.var.M3toM(livestock_withdrawal_m3 * (1 - livestock_water_efficiency))

            # 4. irrigation (surface + reservoir + ground)
            (
                irrigation_water_withdrawal_m,
                irrigation_water_consumption_m,
                return_flow_irrigation_m,
                addtoevapotrans_m,
            ) = self.farmers.abstract_water(
                cell_area = self.var.cellArea.get() if self.model.args.use_gpu else self.var.cellArea,
                landunit_to_var=self.var.landunit_to_var,
                totalPotIrrConsumption=pot_irrConsumption.get() if self.model.args.use_gpu else pot_irrConsumption,
                available_channel_storage_m3=available_channel_storage_m3,
                available_groundwater_m3=available_groundwater_m3,
                groundwater_head=groundwater_head,
                available_reservoir_storage_m3=available_reservoir_storage_m3,
                command_areas=self.var.reservoir_command_areas.get() if self.model.args.use_gpu else self.var.reservoir_command_areas,
            )


            if checkOption('calcWaterBalance'):
                self.model.waterbalance_module.waterBalanceCheck(
                    how='cellwise',
                    influxes=[irrigation_water_withdrawal_m],
                    outfluxes=[irrigation_water_consumption_m, addtoevapotrans_m, return_flow_irrigation_m],
                    tollerance=1e-7
                )
            
            if self.model.args.use_gpu:
                # reservoir_abstraction = cp.asarray(reservoir_abstraction_m)
                ## Water application
                self.var.actual_irrigation_consumption = cp.asarray(irrigation_water_consumption_m)
                addtoevapotrans = cp.asarray(addtoevapotrans_m)
                # TODO: Add other return flows
            else:
                self.var.actual_irrigation_consumption = irrigation_water_consumption_m
                addtoevapotrans = addtoevapotrans_m

            assert (pot_irrConsumption + 1e-7 >= self.var.actual_irrigation_consumption).all()
            assert (self.var.actual_irrigation_consumption + 1e-7 >= 0).all()

            groundwater_abstraction_m3 = available_groundwater_m3_pre - available_groundwater_m3
            channel_abstraction_m3 = available_channel_storage_m3_pre - available_channel_storage_m3
            
            if checkOption('includeWaterBodies'): 
                reservoir_abstraction_m3 = available_reservoir_storage_m3_pre - available_reservoir_storage_m3
                # Abstract water from reservoir
                self.model.data.var.lakeStorageC -= reservoir_abstraction_m3
                self.model.data.var.lakeVolumeM3C -= reservoir_abstraction_m3
                self.model.data.var.lakeResStorageC -= reservoir_abstraction_m3
                self.model.data.var.reservoirStorageM3C -= reservoir_abstraction_m3

            returnFlow = self.model.data.to_var(landunit_data=return_flow_irrigation_m, fn='mean') + domestic_return_flow_m + industry_return_flow_m + livestock_return_flow_m
                
            if checkOption('calcWaterBalance'):
                self.model.waterbalance_module.waterBalanceCheck(
                    how='sum',
                    influxes=[],
                    outfluxes=[
                        domestic_withdrawal_m3,
                        industry_withdrawal_m3,
                        livestock_withdrawal_m3,
                        irrigation_water_withdrawal_m * self.var.cellArea.get() if self.model.args.use_gpu else self.var.cellArea,

                    ],
                    prestorages=[available_channel_storage_m3_pre, available_reservoir_storage_m3_pre, available_groundwater_m3_pre],
                    poststorages=[available_channel_storage_m3, available_reservoir_storage_m3, available_groundwater_m3],
                    tollerance=10000
                )

            return (
                groundwater_abstraction_m3 / self.model.data.var.cellArea,
                channel_abstraction_m3 / self.model.data.var.cellArea,
                addtoevapotrans,
                returnFlow,
            )