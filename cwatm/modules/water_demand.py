# -------------------------------------------------------------------------
# Name:        Waterdemand module
# Purpose:
#
# Author:      PB, JdB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass
import rasterio
from .soil import (
    get_root_ratios,
    get_maximum_water_content,
    get_critical_water_level,
    get_available_water,
    get_fraction_easily_available_soil_water,
    get_crop_group_number,
)

from geb.workflows import TimingModule


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
    waterDemandLost"""

    def __init__(self, model):
        """
        Initial part of the water demand module

        Set the water allocation
        """
        self.model = model
        self.var = model.data.HRU
        self.crop_farmers = model.agents.crop_farmers
        self.livestock_farmers = model.agents.livestock_farmers
        self.industry = model.agents.industry
        self.households = model.agents.households
        self.reservoir_operators = model.agents.reservoir_operators

        with rasterio.open(
            self.model.model_structure["subgrid"][
                "routing/lakesreservoirs/subcommand_areas"
            ],
            "r",
        ) as src:
            reservoir_command_areas = self.var.compress(src.read(1), method="last")
            water_body_mapping = self.model.lakes_reservoirs_module.waterbody_mapping
            self.var.reservoir_command_areas = np.take(
                water_body_mapping, reservoir_command_areas, mode="clip"
            )

        self.model.data.grid.leakageC = np.compress(
            self.model.data.grid.compress_LR,
            self.model.data.grid.full_compressed(0, dtype=np.float32),
        )

    def get_potential_irrigation_consumption(self, totalPotET):
        """Calculate the potential irrigation consumption. Not that consumption
        is not the same as withdrawal. Consumption is the amount of water that
        is actually used by the farmers, while withdrawal is the amount of water
        that is taken from the source. The difference is the return flow."""
        # Paddy irrigation -> No = 2
        # Non paddy irrigation -> No = 3

        # a function of cropKC (evaporation and transpiration) and available water see Wada et al. 2014 p. 19
        paddy_irrigated_land = np.where(self.var.land_use_type == 2)

        paddy_level = self.var.full_compressed(np.nan, dtype=np.float32)
        paddy_level[paddy_irrigated_land] = (
            self.var.topwater[paddy_irrigated_land]
            + self.var.natural_available_water_infiltration[paddy_irrigated_land]
        )

        nonpaddy_irrigated_land = np.where(self.var.land_use_type == 3)[0]

        # load crop group number
        crop_group_number = get_crop_group_number(
            self.var.crop_map,
            self.model.agents.crop_farmers.crop_data["crop_group_number"].values,
            self.var.land_use_type,
            self.var.natural_crop_groups,
        )

        # p is between 0 and 1 => if p =1 wcrit = wwp, if p= 0 wcrit = wfc
        p = get_fraction_easily_available_soil_water(
            crop_group_number[nonpaddy_irrigated_land],
            totalPotET[nonpaddy_irrigated_land],
        )

        root_ratios = get_root_ratios(
            self.var.root_depth[nonpaddy_irrigated_land],
            self.var.soil_layer_height[:, nonpaddy_irrigated_land],
        )

        max_water_content1 = (
            get_maximum_water_content(
                self.var.wfc1[nonpaddy_irrigated_land],
                self.var.wwp1[nonpaddy_irrigated_land],
            )
            * root_ratios[0]
        )
        max_water_content2 = (
            get_maximum_water_content(
                self.var.wfc2[nonpaddy_irrigated_land],
                self.var.wwp2[nonpaddy_irrigated_land],
            )
            * root_ratios[1]
        )
        max_water_content3 = (
            get_maximum_water_content(
                self.var.wfc3[nonpaddy_irrigated_land],
                self.var.wwp3[nonpaddy_irrigated_land],
            )
            * root_ratios[2]
        )
        max_water_content = self.var.full_compressed(np.nan, dtype=np.float32)
        max_water_content[nonpaddy_irrigated_land] = (
            max_water_content1 + max_water_content2 + max_water_content3
        )

        critical_water_level1 = (
            get_critical_water_level(
                p,
                self.var.wfc1[nonpaddy_irrigated_land],
                self.var.wwp1[nonpaddy_irrigated_land],
            )
            * root_ratios[0]
        )
        critical_water_level2 = (
            get_critical_water_level(
                p,
                self.var.wfc2[nonpaddy_irrigated_land],
                self.var.wwp2[nonpaddy_irrigated_land],
            )
            * root_ratios[1]
        )
        critical_water_level3 = (
            get_critical_water_level(
                p,
                self.var.wfc3[nonpaddy_irrigated_land],
                self.var.wwp3[nonpaddy_irrigated_land],
            )
            * root_ratios[2]
        )
        critical_water_level = self.var.full_compressed(np.nan, dtype=np.float32)
        critical_water_level[nonpaddy_irrigated_land] = (
            critical_water_level1 + critical_water_level2 + critical_water_level3
        )

        readily_available_water1 = (
            get_available_water(
                self.var.w1[nonpaddy_irrigated_land],
                self.var.wwp1[nonpaddy_irrigated_land],
            )
            * root_ratios[0]
        )
        readily_available_water2 = (
            get_available_water(
                self.var.w2[nonpaddy_irrigated_land],
                self.var.wwp2[nonpaddy_irrigated_land],
            )
            * root_ratios[1]
        )
        readily_available_water3 = (
            get_available_water(
                self.var.w3[nonpaddy_irrigated_land],
                self.var.wwp3[nonpaddy_irrigated_land],
            )
            * root_ratios[2]
        )
        readily_available_water = self.var.full_compressed(np.nan, dtype=np.float32)
        readily_available_water[nonpaddy_irrigated_land] = (
            readily_available_water1
            + readily_available_water2
            + readily_available_water3
        )

        # first 2 soil layers to estimate distribution between runoff and infiltration
        soilWaterStorage = (
            self.var.w1[nonpaddy_irrigated_land] + self.var.w2[nonpaddy_irrigated_land]
        )
        soilWaterStorageCap = (
            self.var.ws1[nonpaddy_irrigated_land]
            + self.var.ws2[nonpaddy_irrigated_land]
        )
        relSat = soilWaterStorage / soilWaterStorageCap
        satAreaFrac = 1 - (1 - relSat) ** self.var.arnoBeta[nonpaddy_irrigated_land]
        satAreaFrac = np.maximum(np.minimum(satAreaFrac, 1.0), 0.0)

        store = soilWaterStorageCap / (self.var.arnoBeta[nonpaddy_irrigated_land] + 1)
        potBeta = (self.var.arnoBeta[nonpaddy_irrigated_land] + 1) / self.var.arnoBeta[
            nonpaddy_irrigated_land
        ]
        potential_infiltration_capacity = self.var.full_compressed(
            np.nan, dtype=np.float32
        )
        potential_infiltration_capacity[nonpaddy_irrigated_land] = store - store * (
            1 - (1 - satAreaFrac) ** potBeta
        )

        return (
            paddy_level,
            readily_available_water,
            critical_water_level,
            max_water_content,
            potential_infiltration_capacity,
        )

    def get_available_water(self):
        assert (
            self.model.data.grid.waterBodyIDC.size == self.model.data.grid.storage.size
        )
        assert (
            self.model.data.grid.waterBodyIDC.size
            == self.model.data.grid.waterBodyTypC.size
        )
        available_reservoir_storage_m3 = np.zeros_like(self.model.data.grid.storage)
        available_reservoir_storage_m3[self.model.data.grid.waterBodyTypC == 2] = (
            self.reservoir_operators.get_available_water_reservoir_command_areas(
                self.model.data.grid.storage[self.model.data.grid.waterBodyTypC == 2]
            )
        )
        return (
            self.model.data.grid.channelStorageM3.copy(),
            available_reservoir_storage_m3,
            self.model.groundwater_module.groundwater_content_m3,
            self.model.groundwater_module.head,
        )

    def withdraw(self, source, demand):
        withdrawal = np.minimum(source, demand)
        source -= withdrawal  # update in place
        demand -= withdrawal  # update in place
        return withdrawal

    def step(self, totalPotET):
        """
        Dynamic part of the water demand module

        * calculate the fraction of water from surface water vs. groundwater
        * get non-Irrigation water demand and its return flow fraction
        """

        timer = TimingModule("Water demand")

        # WATER DEMAND
        domestic_water_demand, domestic_water_efficiency = (
            self.households.water_demand()
        )
        timer.new_split("Domestic")
        industry_water_demand, industry_water_efficiency = self.industry.water_demand()
        timer.new_split("Industry")
        livestock_water_demand, livestock_water_efficiency = (
            self.livestock_farmers.water_demand()
        )
        timer.new_split("Livestock")
        (
            paddy_level,
            readily_available_water,
            critical_water_level,
            max_water_content,
            potential_infiltration_capacity,
        ) = self.get_potential_irrigation_consumption(totalPotET)

        assert (domestic_water_demand >= 0).all()
        assert (industry_water_demand >= 0).all()
        assert (livestock_water_demand >= 0).all()

        (
            available_channel_storage_m3,
            available_reservoir_storage_m3,
            available_groundwater_m,
            groundwater_head,
        ) = self.get_available_water()

        available_groundwater_m3 = self.model.data.grid.MtoM3(available_groundwater_m)

        available_channel_storage_m3_pre = available_channel_storage_m3.copy()
        available_reservoir_storage_m3_pre = available_reservoir_storage_m3.copy()
        available_groundwater_m3_pre = available_groundwater_m3.copy()

        # water withdrawal
        # 1. domestic (surface + ground)
        domestic_water_demand = self.model.data.to_grid(
            HRU_data=domestic_water_demand, fn="weightedmean"
        )
        domestic_water_demand_m3 = self.model.data.grid.MtoM3(domestic_water_demand)
        del domestic_water_demand

        domestic_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, domestic_water_demand_m3
        )  # withdraw from surface water
        domestic_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, domestic_water_demand_m3
        )  # withdraw from groundwater
        domestic_return_flow_m = self.model.data.grid.M3toM(
            domestic_withdrawal_m3 * (1 - domestic_water_efficiency)
        )

        # 2. industry (surface + ground)
        industry_water_demand = self.model.data.to_grid(
            HRU_data=industry_water_demand, fn="weightedmean"
        )
        industry_water_demand_m3 = self.model.data.grid.MtoM3(industry_water_demand)
        del industry_water_demand

        industry_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, industry_water_demand_m3
        )  # withdraw from surface water
        industry_withdrawal_m3 += self.withdraw(
            available_groundwater_m3, industry_water_demand_m3
        )  # withdraw from groundwater
        industry_return_flow_m = self.model.data.grid.M3toM(
            industry_withdrawal_m3 * (1 - industry_water_efficiency)
        )

        # 3. livestock (surface)
        livestock_water_demand = self.model.data.to_grid(
            HRU_data=livestock_water_demand, fn="weightedmean"
        )
        livestock_water_demand_m3 = self.model.data.grid.MtoM3(livestock_water_demand)
        del livestock_water_demand

        livestock_withdrawal_m3 = self.withdraw(
            available_channel_storage_m3, livestock_water_demand_m3
        )  # withdraw from surface water
        livestock_return_flow_m = self.model.data.grid.M3toM(
            livestock_withdrawal_m3 * (1 - livestock_water_efficiency)
        )
        timer.new_split("Water withdrawal")

        # 4. irrigation (surface + reservoir + ground)
        (
            irrigation_water_withdrawal_m,
            irrigation_water_consumption_m,
            return_flow_irrigation_m,
            addtoevapotrans_m,
        ) = self.crop_farmers.abstract_water(
            cell_area=(
                self.var.cellArea.get() if self.model.use_gpu else self.var.cellArea
            ),
            HRU_to_grid=self.var.HRU_to_grid,
            paddy_level=paddy_level,
            readily_available_water=readily_available_water,
            critical_water_level=critical_water_level,
            max_water_content=max_water_content,
            potential_infiltration_capacity=potential_infiltration_capacity,
            available_channel_storage_m3=available_channel_storage_m3,
            available_groundwater_m3=available_groundwater_m3,
            groundwater_head=groundwater_head,
            groundwater_depth=self.model.groundwater_module.groundwater_depth,
            available_reservoir_storage_m3=available_reservoir_storage_m3,
            command_areas=(
                self.var.reservoir_command_areas.get()
                if self.model.use_gpu
                else self.var.reservoir_command_areas
            ),
        )
        timer.new_split("Irrigation")

        if self.model.CHECK_WATER_BALANCE:
            self.model.waterbalance_module.waterBalanceCheck(
                name="water_demand_1",
                how="cellwise",
                influxes=[irrigation_water_withdrawal_m],
                outfluxes=[
                    irrigation_water_consumption_m,
                    addtoevapotrans_m,
                    return_flow_irrigation_m,
                ],
                tollerance=1e-7,
            )

        if self.model.use_gpu:
            # reservoir_abstraction = cp.asarray(reservoir_abstraction_m)
            ## Water application
            self.var.actual_irrigation_consumption = cp.asarray(
                irrigation_water_consumption_m
            )
            addtoevapotrans = cp.asarray(addtoevapotrans_m)
        else:
            self.var.actual_irrigation_consumption = irrigation_water_consumption_m
            addtoevapotrans = addtoevapotrans_m

        assert (self.var.actual_irrigation_consumption + 1e-6 >= 0).all()

        groundwater_abstraction_m3 = (
            available_groundwater_m3_pre - available_groundwater_m3
        )
        channel_abstraction_m3 = (
            available_channel_storage_m3_pre - available_channel_storage_m3
        )

        reservoir_abstraction_m3 = (
            available_reservoir_storage_m3_pre - available_reservoir_storage_m3
        )
        assert (
            self.model.data.grid.waterBodyTypC[np.where(reservoir_abstraction_m3 > 0)]
            == 2
        ).all()

        # Abstract water from reservoir
        self.model.data.grid.storage -= reservoir_abstraction_m3

        returnFlow = (
            self.model.data.to_grid(
                HRU_data=return_flow_irrigation_m, fn="weightedmean"
            )
            + domestic_return_flow_m
            + industry_return_flow_m
            + livestock_return_flow_m
        )

        if self.model.CHECK_WATER_BALANCE:
            self.model.waterbalance_module.waterBalanceCheck(
                name="water_demand_2",
                how="sum",
                influxes=[],
                outfluxes=[
                    domestic_withdrawal_m3,
                    industry_withdrawal_m3,
                    livestock_withdrawal_m3,
                    (
                        irrigation_water_withdrawal_m * self.var.cellArea.get()
                        if self.model.use_gpu
                        else self.var.cellArea
                    ),
                ],
                prestorages=[
                    available_channel_storage_m3_pre,
                    available_reservoir_storage_m3_pre,
                    available_groundwater_m3_pre,
                ],
                poststorages=[
                    available_channel_storage_m3,
                    available_reservoir_storage_m3,
                    available_groundwater_m3,
                ],
                tollerance=10000,
            )
        if self.model.timing:
            print(timer)

        return (
            groundwater_abstraction_m3 / self.model.data.grid.cellArea,
            channel_abstraction_m3 / self.model.data.grid.cellArea,
            addtoevapotrans,
            returnFlow,
        )
