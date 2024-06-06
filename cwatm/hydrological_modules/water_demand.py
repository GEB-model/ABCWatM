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
from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import (
    option,
    cbinding,
    loadmap,
    checkOption,
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
    waterDemandLost
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.model = model
        self.var = model.data.HRU
        self.crop_farmers = model.agents.crop_farmers
        self.livestock_farmers = model.agents.livestock_farmers
        self.industry = model.agents.industry
        self.households = model.agents.households
        self.reservoir_operators = model.agents.reservoir_operators

    def potential_irrigation_consumption(self, totalPotET):
        pot_irrConsumption = self.var.full_compressed(0, dtype=np.float32)
        # Paddy irrigation -> No = 2
        # Non paddy irrigation -> No = 3

        # a function of cropKC (evaporation and transpiration) and available water see Wada et al. 2014 p. 19
        paddy_irrigated_land = np.where(self.var.land_use_type == 2)
        pot_irrConsumption[paddy_irrigated_land] = np.where(
            self.var.cropKC[paddy_irrigated_land] > 0.75,
            np.maximum(
                0.0,
                (
                    self.var.maxtopwater
                    - (
                        self.var.topwater[paddy_irrigated_land]
                        + self.var.natural_available_water_infiltration[
                            paddy_irrigated_land
                        ]
                    )
                ),
            ),
            0.0,
        )
        assert not np.isnan(pot_irrConsumption).any()

        nonpaddy_irrigated_land = np.where(self.var.land_use_type == 3)[0]

        # Infiltration capacity
        #  ========================================
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
        potInfiltrationCapacity = store - store * (1 - (1 - satAreaFrac) ** potBeta)
        # ----------------------------------------------------------
        availWaterPlant1 = np.maximum(
            0.0,
            self.var.w1[nonpaddy_irrigated_land]
            - self.var.wwp1[nonpaddy_irrigated_land],
        )  # * self.var.rootDepth[0][No]  should not be multiplied again with soildepth
        availWaterPlant2 = np.maximum(
            0.0,
            self.var.w2[nonpaddy_irrigated_land]
            - self.var.wwp2[nonpaddy_irrigated_land],
        )  # * self.var.rootDepth[1][No]
        # availWaterPlant3 = np.maximum(0., self.var.w3[No] - self.var.wwp3[No])  #* self.var.rootDepth[2][No]
        readAvlWater = availWaterPlant1 + availWaterPlant2  # + availWaterPlant3

        # calculate   ****** SOIL WATER STRESS ************************************

        # The crop group number is a indicator of adaptation to dry climate,
        # e.g. olive groves are adapted to dry climate, therefore they can extract more water from drying out soil than e.g. rice.
        # The crop group number of olive groves is 4 and of rice fields is 1
        # for irrigation it is expected that the crop has a low adaptation to dry climate
        # cropGroupNumber = 1.0
        etpotMax = np.minimum(0.1 * (totalPotET[nonpaddy_irrigated_land] * 1000.0), 1.0)
        # to avoid a strange behaviour of the p-formula's, ETRef is set to a maximum of 10 mm/day.

        # for group number 1 -> those are plants which needs irrigation
        # p = 1 / (0.76 + 1.5 * np.minimum(0.1 * (self.var.totalPotET[No] * 1000.), 1.0)) - 0.10 * ( 5 - cropGroupNumber)
        p = 1 / (0.76 + 1.5 * etpotMax) - 0.4
        # soil water depletion fraction (easily available soil water) # Van Diepen et al., 1988: WOFOST 6.0, p.87.
        p = p + (etpotMax - 0.6) / 4
        # correction for crop group 1  (Van Diepen et al, 1988) -> p between 0.14 - 0.77
        p = np.maximum(np.minimum(p, 1.0), 0.0)
        # p is between 0 and 1 => if p =1 wcrit = wwp, if p= 0 wcrit = wfc
        # p is closer to 0 if evapo is bigger and cropgroup is smaller

        wCrit1 = (
            (1 - p)
            * (
                self.var.wfc1[nonpaddy_irrigated_land]
                - self.var.wwp1[nonpaddy_irrigated_land]
            )
        ) + self.var.wwp1[nonpaddy_irrigated_land]
        wCrit2 = (
            (1 - p)
            * (
                self.var.wfc2[nonpaddy_irrigated_land]
                - self.var.wwp2[nonpaddy_irrigated_land]
            )
        ) + self.var.wwp2[nonpaddy_irrigated_land]
        # wCrit3 = ((1 - p) * (self.var.wfc3[No] - self.var.wwp3[No])) + self.var.wwp3[No]

        critWaterPlant1 = np.maximum(
            0.0, wCrit1 - self.var.wwp1[nonpaddy_irrigated_land]
        )  # * self.var.rootDepth[0][No]
        critWaterPlant2 = np.maximum(
            0.0, wCrit2 - self.var.wwp2[nonpaddy_irrigated_land]
        )  # * self.var.rootDepth[1][No]
        # critWaterPlant3 = np.maximum(0., wCrit3 - self.var.wwp3[No]) # * self.var.rootDepth[2][No]
        critAvlWater = critWaterPlant1 + critWaterPlant2  # + critWaterPlant3

        pot_irrConsumption[nonpaddy_irrigated_land] = np.where(
            self.var.cropKC[nonpaddy_irrigated_land] > 0.20,
            np.where(
                readAvlWater < critAvlWater,
                np.maximum(
                    0.0, self.var.totAvlWater[nonpaddy_irrigated_land] - readAvlWater
                ),
                0.0,
            ),
            0.0,
        )
        assert not np.isnan(pot_irrConsumption).any()
        # should not be bigger than infiltration capacity
        pot_irrConsumption[nonpaddy_irrigated_land] = np.minimum(
            pot_irrConsumption[nonpaddy_irrigated_land], potInfiltrationCapacity
        )

        assert not np.isnan(pot_irrConsumption).any()
        return pot_irrConsumption

    def initial(self):
        """
        Initial part of the water demand module

        Set the water allocation
        """

        if checkOption("includeWaterDemand"):
            if checkOption("includeWaterBodies"):
                self.var.using_reservoir_command_areas = False
                if "using_reservoir_command_areas" in option:
                    if checkOption("using_reservoir_command_areas"):
                        self.var.using_reservoir_command_areas = True
                        with rasterio.open(
                            cbinding("reservoir_command_areas"), "r"
                        ) as src:
                            reservoir_command_areas = self.model.data.grid.compress(
                                src.read(1)
                            )
                            water_body_mapping = np.full(
                                self.model.data.grid.waterBodyID.max() + 1,
                                0,
                                dtype=np.int32,
                            )
                            water_body_ids = np.compress(
                                self.model.data.grid.compress_LR,
                                self.model.data.grid.waterBodyID,
                            )
                            water_body_mapping[water_body_ids] = np.arange(
                                0, water_body_ids.size, dtype=np.int32
                            )
                            reservoir_command_areas_mapped = water_body_mapping[
                                reservoir_command_areas
                            ]
                            reservoir_command_areas_mapped[
                                reservoir_command_areas == -1
                            ] = -1
                            self.var.reservoir_command_areas = self.model.data.to_HRU(
                                data=reservoir_command_areas_mapped
                            )

                self.var.using_lift_command_areas = False
                if "using_lift_command_areas" in option:
                    if checkOption("using_lift_command_areas"):
                        self.var.using_lift_command_areas = True
                        lift_command_areas = loadmap("lift_command_areas").astype(
                            np.int
                        )
                        self.var.lift_command_areas = self.model.data.to_HRU(
                            data=lift_command_areas, fn=None
                        )
            else:
                self.var.reservoir_command_areas = self.var.full_compressed(
                    -1, dtype=np.int32
                )

            if checkOption("canal_leakage"):
                self.model.data.grid.leakageC = np.compress(
                    self.model.data.grid.compress_LR,
                    self.model.data.grid.full_compressed(0, dtype=np.float32),
                )

            self.model.data.grid.gwstorage_full = (
                float(cbinding("poro")) * float(cbinding("thickness")) + globals.inZero
            )

    def get_available_water(self, potential_irrigation_consumption_m):
        assert (
            self.model.data.grid.waterBodyIDC.size
            == self.model.data.grid.reservoirStorageM3C.size
        )
        assert (
            self.model.data.grid.waterBodyIDC.size
            == self.model.data.grid.waterBodyTypC.size
        )
        available_reservoir_storage_m3 = np.zeros_like(
            self.model.data.grid.reservoirStorageM3C
        )
        available_reservoir_storage_m3[self.model.data.grid.waterBodyTypC == 2] = (
            self.reservoir_operators.get_available_water_reservoir_command_areas(
                self.model.data.grid.reservoirStorageM3C[
                    self.model.data.grid.waterBodyTypC == 2
                ],
                potential_irrigation_consumption_m,
            )
        )
        return (
            self.model.data.grid.channelStorageM3.copy(),
            available_reservoir_storage_m3,
            self.model.groundwater_modflow_module.available_groundwater_m,
            self.model.data.grid.head,
        )

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

        timer = TimingModule("Water demand")

        if checkOption("includeWaterDemand"):
            # WATER DEMAND
            domestic_water_demand, domestic_water_efficiency = (
                self.households.water_demand()
            )
            timer.new_split("Domestic")
            industry_water_demand, industry_water_efficiency = (
                self.industry.water_demand()
            )
            timer.new_split("Industry")
            livestock_water_demand, livestock_water_efficiency = (
                self.livestock_farmers.water_demand()
            )
            timer.new_split("Livestock")
            pot_irrConsumption = self.potential_irrigation_consumption(totalPotET)

            assert (domestic_water_demand >= 0).all()
            assert (industry_water_demand >= 0).all()
            assert (livestock_water_demand >= 0).all()
            assert (pot_irrConsumption >= 0).all()

            (
                available_channel_storage_m3,
                available_reservoir_storage_m3,
                available_groundwater_m,
                groundwater_head,
            ) = self.get_available_water(pot_irrConsumption)
            available_groundwater_m3 = self.model.data.grid.MtoM3(
                available_groundwater_m
            )

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
            livestock_water_demand_m3 = self.model.data.grid.MtoM3(
                livestock_water_demand
            )
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
                totalPotIrrConsumption=(
                    pot_irrConsumption.get()
                    if self.model.use_gpu
                    else pot_irrConsumption
                ),
                available_channel_storage_m3=available_channel_storage_m3,
                available_groundwater_m3=available_groundwater_m3,
                groundwater_head=groundwater_head,
                groundwater_depth=self.model.data.grid.groundwater_depth,
                available_reservoir_storage_m3=available_reservoir_storage_m3,
                command_areas=(
                    self.var.reservoir_command_areas.get()
                    if self.model.use_gpu
                    else self.var.reservoir_command_areas
                ),
            )
            timer.new_split("Irrigation")

            if checkOption("calcWaterBalance"):
                self.model.waterbalance_module.waterBalanceCheck(
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

            assert (
                pot_irrConsumption + 1e-6 >= self.var.actual_irrigation_consumption
            ).all()
            assert (self.var.actual_irrigation_consumption + 1e-6 >= 0).all()

            groundwater_abstraction_m3 = (
                available_groundwater_m3_pre - available_groundwater_m3
            )
            channel_abstraction_m3 = (
                available_channel_storage_m3_pre - available_channel_storage_m3
            )

            if checkOption("includeWaterBodies"):
                reservoir_abstraction_m3 = (
                    available_reservoir_storage_m3_pre - available_reservoir_storage_m3
                )
                assert (
                    self.model.data.grid.waterBodyTypC[
                        np.where(reservoir_abstraction_m3 > 0)
                    ]
                    == 2
                ).all()
                # print('reservoir_abs_ratio_sum', round(reservoir_abstraction_m3[[self.model.data.grid.waterBodyTypC == 2]].sum() / self.model.data.grid.reservoirStorageM3C[[self.model.data.grid.waterBodyTypC == 2]].sum(), 3))
                reservoir_abstraction_m3[reservoir_abstraction_m3 > 0] = (
                    reservoir_abstraction_m3[reservoir_abstraction_m3 > 0]
                    / self.model.data.grid.area_command_area_in_study_area[
                        reservoir_abstraction_m3 > 0
                    ]
                )
                reservoir_abstraction_m3 = np.minimum(
                    available_reservoir_storage_m3_pre, reservoir_abstraction_m3
                )

                # Abstract water from reservoir
                self.model.data.grid.lakeResStorageC -= reservoir_abstraction_m3
                # assert (self.model.data.grid.lakeResStorageC >= 0).all()
                self.model.data.grid.reservoirStorageM3C -= reservoir_abstraction_m3
                # assert (self.model.data.grid.lakeResStorageC >= 0).all()

                self.model.data.grid.lakeResStorage = (
                    self.model.data.grid.full_compressed(0, dtype=np.float32)
                )
                np.put(
                    self.model.data.grid.lakeResStorage,
                    self.model.data.grid.decompress_LR,
                    self.model.data.grid.lakeResStorageC,
                )

            returnFlow = (
                self.model.data.to_grid(
                    HRU_data=return_flow_irrigation_m, fn="weightedmean"
                )
                + domestic_return_flow_m
                + industry_return_flow_m
                + livestock_return_flow_m
            )

            if checkOption("calcWaterBalance"):
                self.model.waterbalance_module.waterBalanceCheck(
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