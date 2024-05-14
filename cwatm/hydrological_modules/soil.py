# -------------------------------------------------------------------------
# Name:        Soil module
# Purpose:
#
# Author:      PB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016 based on PCRGLOBE, LISFLOOD, HBV
# -------------------------------------------------------------------------

import numpy as np
from cwatm.management_modules.data_handling import loadmap, divideValues, checkOption
from pathlib import Path


class soil(object):
    """
    **SOIL**


    Calculation vertical transfer of water based on Arno scheme


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    modflow               Flag: True if modflow_coupling = True in settings file                            --
    storGroundwater       simulated groundwater storage                                                     m
    capRiseFrac           fraction of a grid cell where capillar rise may happen                            m
    cropKC                crop coefficient for each of the 4 different land cover types (forest, irrigated  --
    EWRef                 potential evaporation rate from water surface                                     m
    capillar              Simulated flow from groundwater to the third CWATM soil layer                     m
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m
    potTranspiration      Potential transpiration (after removing of evaporation)                           m
    actualET              simulated evapotranspiration from soil, flooded area and vegetation               m
    soilLayers            Number of soil layers                                                             --
    fracVegCover          Fraction of area covered by the corresponding landcover type
    rootDepth
    soildepth             Thickness of the first soil layer                                                 m
    soildepth12           Total thickness of layer 2 and 3                                                  m
    KSat1
    KSat2
    KSat3
    genuM1
    genuM2
    genuM3
    genuInvM1
    genuInvM2
    genuInvM3
    ws1                   Maximum storage capacity in layer 1                                               m
    ws2                   Maximum storage capacity in layer 2                                               m
    ws3                   Maximum storage capacity in layer 3                                               m
    wres1                 Residual storage capacity in layer 1                                              m
    wres2                 Residual storage capacity in layer 2                                              m
    wres3                 Residual storage capacity in layer 3                                              m
    wrange1
    wrange2
    wrange3
    wfc1                  Soil moisture at field capacity in layer 1
    wfc2                  Soil moisture at field capacity in layer 2
    wfc3                  Soil moisture at field capacity in layer 3
    wwp1                  Soil moisture at wilting point in layer 1
    wwp2                  Soil moisture at wilting point in layer 2
    wwp3                  Soil moisture at wilting point in layer 3
    kunSatFC12
    kunSatFC23
    w1                    Simulated water storage in the layer 1                                            m
    w2                    Simulated water storage in the layer 2                                            m
    w3                    Simulated water storage in the layer 3                                            m
    topwater              quantity of water above the soil (flooding)                                       m
    arnoBeta
    adjRoot
    maxtopwater           maximum heigth of topwater                                                        m
    directRunoff          Simulated surface runoff                                                          m
    interflow             Simulated flow reaching runoff instead of groundwater                             m
    openWaterEvap         Simulated evaporation from open areas                                             m
    actTransTotal         Total actual transpiration from the three soil layers                             m
    actBareSoilEvap       Simulated evaporation from the first soil layer                                   m
    FrostIndexThreshold   Degree Days Frost Threshold (stops infiltration, percolation and capillary rise)  --
    FrostIndex            FrostIndex - Molnau and Bissel (1983), A Continuous Frozen Ground Index for Floo  --
    percolationImp        Fraction of area covered by the corresponding landcover type                      m
    cropGroupNumber       soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Dooren  --
    cPrefFlow             Factor influencing preferential flow (flow from surface to GW)                    --
    act_irrConsumption    actual irrgation water consumption                                                m
    potBareSoilEvap       potential bare soil evaporation (calculated with minus snow evaporation)          m
    totalPotET            Potential evaporation per land use class                                          m
    rws                   Transpiration reduction factor (in case of water stress)                          --
    prefFlow              Flow going directly from rainfall to groundwater                                  m
    infiltration          Water actually infiltrating the soil                                              m
    capRiseFromGW         Simulated capillary rise from groundwater                                         m
    NoSubSteps            Number of sub steps to calculate soil percolation                                 --
    perc1to2              Simulated water flow from soil layer 1 to soil layer 2                            m
    perc2to3              Simulated water flow from soil layer 2 to soil layer 3                            m
    perc3toGW             Simulated water flow from soil layer 3 to groundwater                             m
    theta1                fraction of water in soil compartment 1 for each land use class                   --
    theta2                fraction of water in soil compartment 2 for each land use class                   --
    theta3                fraction of water in soil compartment 3 for each land use class                   --
    actTransTotal_forest
    actTransTotal_grassl
    actTransTotal_paddy
    actTransTotal_nonpad
    before
    gwRecharge            groundwater recharge                                                              m
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

    def initial(self):
        """
        Initial part of the soil module

        * Initialize all the hydraulic properties of soil
        * Set soil depth

        """

        # self.var.permeability = float(cbinding('permeability'))

        self.var.soilLayers = 3
        # --- Topography -----------------------------------------------------

        # Fraction of area where percolation to groundwater is impeded [dimensionless]
        self.var.percolationImp = self.model.data.to_HRU(
            data=np.maximum(
                0,
                np.minimum(1, loadmap("percolationImp") * loadmap("factor_interflow")),
            ),
            fn=None,
        )  # checked

        # ------------ Preferential Flow constant ------------------------------------------
        self.var.cropGroupNumber = loadmap("cropgroupnumber")
        self.var.cropGroupNumber = self.model.data.to_HRU(
            data=self.var.cropGroupNumber, fn=None
        )  # checked
        # soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Doorenbos et. al 1978
        # crop groups for formular in van Diepen et al, 1988

        # ------------ Preferential Flow constant ------------------------------------------
        self.var.cPrefFlow = self.model.data.to_HRU(
            data=loadmap("preferentialFlowConstant"), fn=None
        )

        # ------------ SOIL DEPTH ----------------------------------------------------------
        # soil thickness and storage

        self.var.soildepth = np.tile(
            self.var.full_compressed(np.nan, dtype=np.float32), (3, 1)
        )

        # first soil layer = 5 cm
        self.var.soildepth[0] = self.var.full_compressed(0.05, dtype=np.float32)
        # second soil layer minimum 5cm
        stordepth1 = self.model.data.to_HRU(data=loadmap("StorDepth1"), fn=None)
        self.var.soildepth[1] = np.maximum(0.05, stordepth1 - self.var.soildepth[0])

        stordepth2 = self.model.data.to_HRU(data=loadmap("StorDepth2"), fn=None)
        self.var.soildepth[2] = np.maximum(0.05, stordepth2)

        # Calibration
        soildepth_factor = loadmap("soildepth_factor")
        self.var.soildepth[1] = self.var.soildepth[1] * soildepth_factor
        self.var.soildepth[2] = self.var.soildepth[2] * soildepth_factor

        self.model.data.grid.soildepth_12 = self.model.data.to_grid(
            HRU_data=self.var.soildepth[1] + self.var.soildepth[2], fn="mean"
        )

        def create_ini(yaml, idx, plantFATE_cluster, biodiversity_scenario):
            out_dir = self.model.simulation_root / "plantFATE" / f"cell_{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            ini_file = out_dir / f"p_daily.ini"

            yaml["> STRINGS"]["outDir"] = out_dir
            if self.model.scenario in ("spinup", "pre-spinup"):
                original_state_file = (
                    Path("input")
                    / "plantFATE_initialization"
                    / biodiversity_scenario
                    / f"cluster_{plantFATE_cluster}"
                    / "pf_saved_state.txt"
                )
                assert original_state_file.exists()
                new_state_file = out_dir / "pf_saved_state_initialization.txt"
                with open(original_state_file, "r") as original_f:
                    state = original_f.read()
                    timetuple = self.model.current_time.timetuple()
                    year = timetuple.tm_year
                    day_of_year = timetuple.tm_yday
                    state = state.replace(
                        "6 2 0 2000 1 0 0",
                        f"6 2 0 {year + (day_of_year - 1) / 365} 1 0 0",
                    )
                    with open(new_state_file, "w") as new_f:
                        new_f.write(state)

                yaml["> STRINGS"]["continueFromState"] = new_state_file
                yaml["> STRINGS"]["continueFromConfig"] = ini_file
                yaml["> STRINGS"]["saveState"] = True
                yaml["> STRINGS"]["savedStateFile"] = "pf_saved_state_spinup.txt"
                yaml["> STRINGS"]["savedConfigFile"] = "pf_saved_config_spinup.txt"
            else:
                yaml["> STRINGS"]["continueFromState"] = (
                    out_dir
                    / self.model.config["plantFATE"]["> STRINGS"]["exptName"]
                    / "pf_saved_state_spinup.txt"
                )
                yaml["> STRINGS"]["continueFromConfig"] = ini_file
                yaml["> STRINGS"]["savedStateFile"] = None
                yaml["> STRINGS"]["saveState"] = False
                yaml["> STRINGS"]["savedConfigFile"] = None

            with open(ini_file, "w") as f:
                for section, section_dict in yaml.items():
                    f.write(section + "\n")
                    if section_dict is None:
                        continue
                    for key, value in section_dict.items():
                        if value is None:
                            value = "null"
                        elif value is False:
                            value = "no"
                        elif value is True:
                            value = "yes"
                        f.write(key + " " + str(value) + "\n")
            return ini_file

        if self.model.config["general"]["simulate_forest"]:
            plantFATE_cluster = 7
            biodiversity_scenario = "low"

            lon, lat = 73.5975501619, 19.1444726274

            from honeybees.library.raster import coord_to_pixel

            px, py = coord_to_pixel(np.array([lon, lat]), gt=self.model.data.grid.gt)

            cell_ids = np.arange(self.model.data.grid.compressed_size)
            cell_ids_map = self.model.data.grid.decompress(cell_ids, fillvalue=-1)
            cell_id = cell_ids_map[py, px]

            already_has_plantFATE_cell = False
            from cwatm.hydrological_modules import plantFATE

            self.model.plantFATE = []
            self.plantFATE_forest_RUs = np.zeros_like(
                self.var.land_use_type, dtype=bool
            )
            for i, land_use_type_RU in enumerate(self.var.land_use_type):
                grid_cell = self.var.HRU_to_grid[i]
                # if land_use_type_RU == 0 and self.var.land_use_ratio[i] > 0.5:
                if land_use_type_RU == 0 and grid_cell == cell_id:
                    if already_has_plantFATE_cell:
                        self.model.plantFATE.append(None)
                    else:
                        self.plantFATE_forest_RUs[i] = True

                        ini_path = create_ini(
                            self.model.config["plantFATE"],
                            i,
                            plantFATE_cluster,
                            biodiversity_scenario,
                        )
                        already_has_plantFATE_cell = True
                        self.model.plantFATE.append(plantFATE.Model(ini_path))
                else:
                    self.model.plantFATE.append(None)
        return None

    def calculate_soil_water_potential_MPa(
        self,
        soil_moisture,  # [m]
        soil_moisture_wilting_point,  # [m]
        soil_moisture_field_capacity,  # [m]
        soil_tickness,  # [m]
        wilting_point=-1500,  # kPa
        field_capacity=-33,  # kPa
    ):
        # https://doi.org/10.1016/B978-0-12-374460-9.00007-X (eq. 7.16)
        soil_moisture_fraction = soil_moisture / soil_tickness
        # assert (soil_moisture_fraction >= 0).all() and (soil_moisture_fraction <= 1).all()
        del soil_moisture
        soil_moisture_wilting_point_fraction = (
            soil_moisture_wilting_point / soil_tickness
        )
        # assert (soil_moisture_wilting_point_fraction).all() >= 0 and (
        #     soil_moisture_wilting_point_fraction
        # ).all() <= 1
        del soil_moisture_wilting_point
        soil_moisture_field_capacity_fraction = (
            soil_moisture_field_capacity / soil_tickness
        )
        # assert (soil_moisture_field_capacity_fraction >= 0).all() and (
        #     soil_moisture_field_capacity_fraction <= 1
        # ).all()
        del soil_moisture_field_capacity

        n_potential = -(
            np.log(wilting_point / field_capacity)
            / np.log(
                soil_moisture_wilting_point_fraction
                / soil_moisture_field_capacity_fraction
            )
        )
        # assert (n_potential >= 0).all()
        a_potential = 1.5 * 10**6 * soil_moisture_wilting_point_fraction**n_potential
        # assert (a_potential >= 0).all()
        soil_water_potential = -a_potential * soil_moisture_fraction ** (-n_potential)
        return soil_water_potential / 1_000_000  # Pa to MPa

    def calculate_vapour_pressure_deficit_kPa(self, temperature_K, relative_humidity):
        temperature_C = temperature_K - 273.15
        assert (
            temperature_C < 100
        ).all()  # temperature is in Celsius. So on earth should be well below 100.
        assert (
            temperature_C > -100
        ).all()  # temperature is in Celsius. So on earth should be well above -100.
        assert (
            relative_humidity >= 1
        ).all()  # below 1 is so rare that it shouldn't be there at the resolutions of current climate models, and this catches errors with relative_humidity as a ratio [0-1].
        assert (
            relative_humidity <= 100
        ).all()  # below 1 is so rare that it shouldn't be there at the resolutions of current climate models, and this catches errors with relative_humidity as a ratio [0-1].
        # https://soilwater.github.io/pynotes-agriscience/notebooks/vapor_pressure_deficit.html
        saturated_vapour_pressure = 0.611 * np.exp(
            (17.502 * temperature_C) / (temperature_C + 240.97)
        )  # kPa
        actual_vapour_pressure = (
            saturated_vapour_pressure * relative_humidity / 100
        )  # kPa
        vapour_pressure_deficit = saturated_vapour_pressure - actual_vapour_pressure
        return vapour_pressure_deficit

    def calculate_photosynthetic_photon_flux_density(self, shortwave_radiation, xi=0.5):
        # https://search.r-project.org/CRAN/refmans/bigleaf/html/Rg.to.PPFD.html
        photosynthetically_active_radiation = shortwave_radiation * xi
        photosynthetic_photon_flux_density = (
            photosynthetically_active_radiation * 4.6
        )  #  W/m2 -> umol/m2/s
        return photosynthetic_photon_flux_density

    def dynamic(
        self, capillar, openWaterEvap, potTranspiration, potBareSoilEvap, totalPotET
    ):
        """
        Dynamic part of the soil module

        For each of the land cover classes the vertical water transport is simulated
        Distribution of water holding capiacity in 3 soil layers based on saturation excess overland flow, preferential flow
        Dependend on soil depth, soil hydraulic parameters
        """

        from time import time

        t0 = time()

        if checkOption("calcWaterBalance"):
            w1_pre = self.var.w1.copy()
            w2_pre = self.var.w2.copy()
            w3_pre = self.var.w3.copy()
            topwater_pre = self.var.topwater.copy()

        bioarea = np.where(self.var.land_use_type < 4)[0].astype(np.int32)
        paddy_irrigated_land = np.where(self.var.land_use_type == 2)
        irrigated_land = np.where(
            (self.var.land_use_type == 2) | (self.var.land_use_type == 3)
        )
        availWaterInfiltration = (
            self.var.natural_available_water_infiltration
            + self.var.actual_irrigation_consumption
        )
        assert (availWaterInfiltration + 1e-6 >= 0).all()
        availWaterInfiltration[availWaterInfiltration < 0] = 0

        # depending on the crop calender -> here if cropKC > 0.75 paddies are flooded to 50mm (as set in settings file)

        self.var.topwater[paddy_irrigated_land] = np.where(
            self.var.cropKC[paddy_irrigated_land] > 0.75,
            self.var.topwater[paddy_irrigated_land]
            + availWaterInfiltration[paddy_irrigated_land],
            self.var.topwater[paddy_irrigated_land],
        )

        # open water evaporation from the paddy field  - using potential evaporation from open water
        openWaterEvap[paddy_irrigated_land] = np.minimum(
            np.maximum(0.0, self.var.topwater[paddy_irrigated_land]),
            self.var.EWRef[paddy_irrigated_land],
        )
        self.var.topwater[paddy_irrigated_land] = (
            self.var.topwater[paddy_irrigated_land]
            - openWaterEvap[paddy_irrigated_land]
        )

        assert (self.var.topwater >= 0).all()

        # if paddies are flooded, avail water is calculated before: top + avail, otherwise it is calculated here
        availWaterInfiltration[paddy_irrigated_land] = np.where(
            self.var.cropKC[paddy_irrigated_land] > 0.75,
            self.var.topwater[paddy_irrigated_land],
            self.var.topwater[paddy_irrigated_land]
            + availWaterInfiltration[paddy_irrigated_land],
        )

        # open water can evaporate more than maximum bare soil + transpiration because it is calculated from open water pot evaporation
        potBareSoilEvap[paddy_irrigated_land] = np.maximum(
            0.0,
            potBareSoilEvap[paddy_irrigated_land] - openWaterEvap[paddy_irrigated_land],
        )
        # if open water revaporation is bigger than bare soil, transpiration rate is reduced

        ### if GW capillary rise saturates soil layers, water is sent to the above layer, then to runoff
        self.var.w3[bioarea] = self.var.w3[bioarea] + capillar[bioarea]
        # CAPRISE from GW to soilayer 3 , if this is full it is send to soil layer 2
        self.var.w2[bioarea] = self.var.w2[bioarea] + np.where(
            self.var.w3[bioarea] > self.var.ws3[bioarea],
            self.var.w3[bioarea] - self.var.ws3[bioarea],
            0,
        )
        self.var.w3[bioarea] = np.minimum(self.var.ws3[bioarea], self.var.w3[bioarea])
        # CAPRISE from GW to soilayer 2 , if this is full it is send to soil layer 1
        self.var.w1[bioarea] = self.var.w1[bioarea] + np.where(
            self.var.w2[bioarea] > self.var.ws2[bioarea],
            self.var.w2[bioarea] - self.var.ws2[bioarea],
            0,
        )
        self.var.w2[bioarea] = np.minimum(self.var.ws2[bioarea], self.var.w2[bioarea])
        # CAPRISE from GW to soilayer 1 , if this is full it is send to saverunofffromGW
        saverunofffromGW = np.where(
            self.var.w1[bioarea] > self.var.ws1[bioarea],
            self.var.w1[bioarea] - self.var.ws1[bioarea],
            0,
        )
        self.var.w1[bioarea] = np.minimum(self.var.ws1[bioarea], self.var.w1[bioarea])

        assert (self.var.w1 >= 0).all()
        assert (self.var.w2 >= 0).all()
        assert (self.var.w3 >= 0).all()

        # ---------------------------------------------------------
        # calculate transpiration
        # ***** SOIL WATER STRESS ************************************

        etpotMax = np.minimum(0.1 * (totalPotET * 1000.0), 1.0)
        # to avoid a strange behaviour of the p-formula's, ETRef is set to a maximum of 10 mm/day.

        p = self.var.full_compressed(np.nan, dtype=np.float32)

        # for irrigated land
        p[irrigated_land] = 1 / (0.76 + 1.5 * etpotMax[irrigated_land]) - 0.4
        # soil water depletion fraction (easily available soil water) # Van Diepen et al., 1988: WOFOST 6.0, p.87.
        p[irrigated_land] = p[irrigated_land] + (etpotMax[irrigated_land] - 0.6) / 4
        # correction for crop group 1  (Van Diepen et al, 1988) -> p between 0.14 - 0.77
        # The crop group number is a indicator of adaptation to dry climate,
        # e.g. olive groves are adapted to dry climate, therefore they can extract more water from drying out soil than e.g. rice.
        # The crop group number of olive groves is 4 and of rice fields is 1
        # for irrigation it is expected that the crop has a low adaptation to dry climate

        # for non-irrigated bioland
        non_irrigated_bioland = np.where(
            (self.var.land_use_type == 0) | (self.var.land_use_type == 1)
        )
        p[non_irrigated_bioland] = 1 / (
            0.76 + 1.5 * etpotMax[non_irrigated_bioland]
        ) - 0.10 * (5 - self.var.cropGroupNumber[non_irrigated_bioland])
        # soil water depletion fraction (easily available soil water)
        # Van Diepen et al., 1988: WOFOST 6.0, p.87
        # to avoid a strange behaviour of the p-formula's, ETRef is set to a maximum of
        # 10 mm/day. Thus, p will range from 0.15 to 0.45 at ETRef eq 10 and
        # CropGroupNumber 1-5
        p[non_irrigated_bioland] = np.where(
            self.var.cropGroupNumber[non_irrigated_bioland] <= 2.5,
            p[non_irrigated_bioland]
            + (etpotMax[non_irrigated_bioland] - 0.6)
            / (
                self.var.cropGroupNumber[non_irrigated_bioland]
                * (self.var.cropGroupNumber[non_irrigated_bioland] + 3)
            ),
            p[non_irrigated_bioland],
        )
        del non_irrigated_bioland
        del etpotMax
        # correction for crop groups 1 and 2 (Van Diepen et al, 1988)

        p = np.maximum(np.minimum(p, 1.0), 0.0)[bioarea]
        # p is between 0 and 1 => if p =1 wcrit = wwp, if p= 0 wcrit = wfc
        # p is closer to 0 if evapo is bigger and cropgroup is smaller

        wCrit1 = (
            (1 - p) * (self.var.wfc1[bioarea] - self.var.wwp1[bioarea])
        ) + self.var.wwp1[bioarea]
        wCrit2 = (
            (1 - p) * (self.var.wfc2[bioarea] - self.var.wwp2[bioarea])
        ) + self.var.wwp2[bioarea]
        wCrit3 = (
            (1 - p) * (self.var.wfc3[bioarea] - self.var.wwp3[bioarea])
        ) + self.var.wwp3[bioarea]

        del p

        # Transpiration reduction factor (in case of water stress)
        rws1 = divideValues(
            (self.var.w1[bioarea] - self.var.wwp1[bioarea]),
            (wCrit1 - self.var.wwp1[bioarea]),
            default=1.0,
        )
        rws2 = divideValues(
            (self.var.w2[bioarea] - self.var.wwp2[bioarea]),
            (wCrit2 - self.var.wwp2[bioarea]),
            default=1.0,
        )
        rws3 = divideValues(
            (self.var.w3[bioarea] - self.var.wwp3[bioarea]),
            (wCrit3 - self.var.wwp3[bioarea]),
            default=1.0,
        )
        del wCrit1
        del wCrit2
        del wCrit3

        rws1 = np.maximum(np.minimum(1.0, rws1), 0.0) * self.var.adjRoot[0][bioarea]
        rws2 = np.maximum(np.minimum(1.0, rws2), 0.0) * self.var.adjRoot[1][bioarea]
        rws3 = np.maximum(np.minimum(1.0, rws3), 0.0) * self.var.adjRoot[2][bioarea]

        potTranspiration[self.var.indicesDeciduous] = potTranspiration[self.var.indicesDeciduous] * 0.655
        potTranspiration[self.var.indicesConifer] = potTranspiration[self.var.indicesConifer] * 0.84
        potTranspiration[self.var.indicesMixed] = potTranspiration[self.var.indicesMixed] * 0.735

        TaMax = potTranspiration[bioarea] * (rws1 + rws2 + rws3)

        del potTranspiration
        del rws1
        del rws2
        del rws3

        # transpiration is 0 when soil is frozen
        TaMax = np.where(
            self.var.FrostIndex[bioarea] > self.var.FrostIndexThreshold, 0.0, TaMax
        )

        self.model.data.grid.vapour_pressure_deficit = (
            self.calculate_vapour_pressure_deficit_kPa(
                self.model.data.grid.tas, self.model.data.grid.hurs
            )
        )

        self.model.data.grid.photosynthetic_photon_flux_density = (
            self.calculate_photosynthetic_photon_flux_density(self.model.data.grid.rsds)
        )

        soil_water_potential = self.calculate_soil_water_potential_MPa(
            self.var.w1 + self.var.w2 + self.var.w3,  # [m]
            self.var.wwp1 + self.var.wwp2 + self.var.wwp3,  # [m]
            self.var.wfc1 + self.var.wfc2 + self.var.wfc3,  # [m]
            self.var.rootDepth1 + self.var.rootDepth2 + self.var.rootDepth3,  # [m]
            wilting_point=-1500,  # kPa
            field_capacity=-33,  # kPa
        )
        soil_water_potential_plantFATE_HRUs = np.where(
            self.var.land_use_type == 0,
            soil_water_potential,
            np.nan,
        )
        self.model.data.grid.soil_water_potential = self.model.data.to_grid(
            HRU_data=soil_water_potential_plantFATE_HRUs, fn="nanmean"
        )

        if self.model.config["general"]["simulate_forest"]:
            transpiration_plantFATE = np.zeros_like(
                self.plantFATE_forest_RUs, dtype=np.float32
            )  # transpiration in a hydrological model is transpiration from plants and evaporation from the plant's surface in plantFATE.
            # soil_specific_depletion_1_plantFATE = np.zeros_like(self.plantFATE_forest_RUs, dtype=np.float32)
            # soil_specific_depletion_2_plantFATE = np.zeros_like(self.plantFATE_forest_RUs, dtype=np.float32)
            # soil_specific_depletion_3_plantFATE = np.zeros_like(self.plantFATE_forest_RUs, dtype=np.float32)

            for forest_RU_idx, is_simulated_by_plantFATE in enumerate(
                self.plantFATE_forest_RUs
            ):
                if is_simulated_by_plantFATE:
                    forest_grid = self.var.HRU_to_grid[forest_RU_idx]

                    plantFATE_data = {
                        "vapour_pressure_deficit": self.model.data.grid.vapour_pressure_deficit[
                            forest_grid
                        ],
                        "soil_water_potential": self.model.data.grid.soil_water_potential[
                            forest_grid
                        ],
                        "photosynthetic_photon_flux_density": self.model.data.grid.photosynthetic_photon_flux_density[
                            forest_grid
                        ],
                        "temperature": self.model.data.grid.tas[forest_grid],
                    }

                    if (
                        self.model.current_timestep
                        == 1
                        # and self.model.scenario == "spinup"
                    ):
                        self.model.plantFATE[forest_RU_idx].first_step(
                            tstart=self.model.current_time, **plantFATE_data
                        )
                        transpiration_plantFATE[forest_RU_idx], _, _, _ = (
                            0,
                            0,
                            0,
                            0,
                        )  # first timestep, set all to 0. Just for initialization of spinup.
                    else:
                        (
                            transpiration_plantFATE[forest_RU_idx],
                            _,
                            _,
                            _,
                        ) = self.model.plantFATE[forest_RU_idx].step(**plantFATE_data)

            ta1 = np.maximum(
                np.minimum(
                    TaMax * self.var.adjRoot[0][bioarea],
                    self.var.w1[bioarea] - self.var.wwp1[bioarea],
                ),
                0.0,
            )
            ta2 = np.maximum(
                np.minimum(
                    TaMax * self.var.adjRoot[1][bioarea],
                    self.var.w2[bioarea] - self.var.wwp2[bioarea],
                ),
                0.0,
            )
            ta3 = np.maximum(
                np.minimum(
                    TaMax * self.var.adjRoot[2][bioarea],
                    self.var.w3[bioarea] - self.var.wwp3[bioarea],
                ),
                0.0,
            )

            CWatM_w_in_plantFATE_cells = (
                self.var.w1[self.plantFATE_forest_RUs]
                + self.var.w2[self.plantFATE_forest_RUs]
                + self.var.w3[self.plantFATE_forest_RUs]
            )

            print(
                "mean transpiration plantFATE",
                transpiration_plantFATE[self.plantFATE_forest_RUs].mean(),
            )

            print(
                "mean transpiration CwatM",
                TaMax[self.plantFATE_forest_RUs[bioarea]].mean(),
            )

            # bioarea_forest = self.plantFATE_forest_RUs[bioarea]
            # ta1[bioarea_forest] = (
            #     self.var.w1[self.plantFATE_forest_RUs]
            #     / CWatM_w_in_plantFATE_cells
            #     * transpiration_plantFATE[self.plantFATE_forest_RUs]
            # )
            # ta2[bioarea_forest] = (
            #     self.var.w2[self.plantFATE_forest_RUs]
            #     / CWatM_w_in_plantFATE_cells
            #     * transpiration_plantFATE[self.plantFATE_forest_RUs]
            # )
            # ta3[bioarea_forest] = (
            #     self.var.w3[self.plantFATE_forest_RUs]
            #     / CWatM_w_in_plantFATE_cells
            #     * transpiration_plantFATE[self.plantFATE_forest_RUs]
            # )

            # assert self.model.waterbalance_module.waterBalanceCheck(
            #     how="cellwise",
            #     influxes=[
            #         ta1[bioarea_forest],
            #         ta2[bioarea_forest],
            #         ta3[bioarea_forest],
            #     ],
            #     outfluxes=[transpiration_plantFATE[self.plantFATE_forest_RUs]],
            #     tollerance=1e-7,
            # )

        else:
            ta1 = np.maximum(
                np.minimum(
                    TaMax * self.var.adjRoot[0][bioarea],
                    self.var.w1[bioarea] - self.var.wwp1[bioarea],
                ),
                0.0,
            )
            ta2 = np.maximum(
                np.minimum(
                    TaMax * self.var.adjRoot[1][bioarea],
                    self.var.w2[bioarea] - self.var.wwp2[bioarea],
                ),
                0.0,
            )
            ta3 = np.maximum(
                np.minimum(
                    TaMax * self.var.adjRoot[2][bioarea],
                    self.var.w3[bioarea] - self.var.wwp3[bioarea],
                ),
                0.0,
            )

        del TaMax

        self.var.w1[bioarea] = self.var.w1[bioarea] - ta1
        self.var.w2[bioarea] = self.var.w2[bioarea] - ta2
        self.var.w3[bioarea] = self.var.w3[bioarea] - ta3

        assert (self.var.w1 >= 0).all()
        assert (self.var.w2 >= 0).all()
        assert (self.var.w3 >= 0).all()

        self.var.actTransTotal[bioarea] = ta1 + ta2 + ta3

        del ta1
        del ta2
        del ta3

        # Actual potential bare soil evaporation - upper layer
        self.var.actBareSoilEvap[bioarea] = np.minimum(
            potBareSoilEvap[bioarea],
            np.maximum(0.0, self.var.w1[bioarea] - self.var.wres1[bioarea]),
        )
        del potBareSoilEvap
        self.var.actBareSoilEvap[bioarea] = np.where(
            self.var.FrostIndex[bioarea] > self.var.FrostIndexThreshold,
            0.0,
            self.var.actBareSoilEvap[bioarea],
        )

        # no bare soil evaporation in the inundated paddy field
        self.var.actBareSoilEvap[paddy_irrigated_land] = np.where(
            self.var.topwater[paddy_irrigated_land] > 0.0,
            0.0,
            self.var.actBareSoilEvap[paddy_irrigated_land],
        )

        self.var.w1[bioarea] = self.var.w1[bioarea] - self.var.actBareSoilEvap[bioarea]

        # Infiltration capacity
        #  ========================================
        # first 2 soil layers to estimate distribution between runoff and infiltration
        soilWaterStorage = self.var.w1[bioarea] + self.var.w2[bioarea]
        soilWaterStorageCap = self.var.ws1[bioarea] + self.var.ws2[bioarea]
        relSat = soilWaterStorage / soilWaterStorageCap
        relSat = np.minimum(relSat, 1.0)

        del soilWaterStorage

        satAreaFrac = 1 - (1 - relSat) ** self.var.arnoBeta[bioarea]
        # Fraction of pixel that is at saturation as a function of
        # the ratio Theta1/ThetaS1. Distribution function taken from
        # Zhao,1977, as cited in Todini, 1996 (JoH 175, 339-382) Eq. A.4.
        satAreaFrac = np.maximum(np.minimum(satAreaFrac, 1.0), 0.0)

        store = soilWaterStorageCap / (self.var.arnoBeta[bioarea] + 1)
        potBeta = (self.var.arnoBeta[bioarea] + 1) / self.var.arnoBeta[bioarea]
        potInf = store - store * (1 - (1 - satAreaFrac) ** potBeta)

        infiltration_multiplier = (
            self.model.agents.farmers.infiltration_multiplier.by_field(
                self.model.data.HRU.land_owners, nofieldvalue=1
            )
        )
        potInf *= infiltration_multiplier[bioarea]

        del satAreaFrac
        del potBeta
        del store
        del soilWaterStorageCap

        # ------------------------------------------------------------------
        # calculate preferential flow
        prefFlow = self.var.full_compressed(0, dtype=np.float32)
        prefFlow[bioarea] = availWaterInfiltration[bioarea] * relSat**self.var.cPrefFlow
        prefFlow[bioarea] = np.where(
            self.var.FrostIndex[bioarea] > self.var.FrostIndexThreshold,
            0.0,
            prefFlow[bioarea],
        )
        prefFlow[paddy_irrigated_land] = 0

        del relSat

        prefFlow[bioarea] = prefFlow[bioarea] * (1 - self.var.capriseindex[bioarea])

        # ---------------------------------------------------------
        # calculate infiltration
        # infiltration, limited with KSat1 and available water in topWaterLayer
        infiltration = self.var.full_compressed(0, dtype=np.float32)
        infiltration[bioarea] = np.minimum(
            potInf, availWaterInfiltration[bioarea] - prefFlow[bioarea]
        )
        del potInf
        infiltration[bioarea] = np.where(
            self.var.FrostIndex[bioarea] > self.var.FrostIndexThreshold,
            0.0,
            infiltration[bioarea],
        )

        directRunoff = self.var.full_compressed(0, dtype=np.float32)
        directRunoff[bioarea] = np.maximum(
            0.0,
            availWaterInfiltration[bioarea] - infiltration[bioarea] - prefFlow[bioarea],
        )

        del availWaterInfiltration

        self.var.topwater[paddy_irrigated_land] = np.maximum(
            0.0,
            self.var.topwater[paddy_irrigated_land]
            - infiltration[paddy_irrigated_land],
        )
        # if paddy fields flooded only runoff if topwater > 0.05m
        h = np.maximum(
            0.0, self.var.topwater[paddy_irrigated_land] - self.var.maxtopwater
        )
        directRunoff[paddy_irrigated_land] = np.where(
            self.var.cropKC[paddy_irrigated_land] > 0.75,
            h,
            directRunoff[paddy_irrigated_land],
        )
        del h
        self.var.topwater[paddy_irrigated_land] = np.maximum(
            0.0,
            self.var.topwater[paddy_irrigated_land]
            - directRunoff[paddy_irrigated_land],
        )

        directRunoff[bioarea] = directRunoff[bioarea] + saverunofffromGW
        # ADDING EXCESS WATER FROM GW CAPILLARY RISE

        del saverunofffromGW

        # infiltration to soilayer 1 , if this is full it is send to soil layer 2
        self.var.w1[bioarea] = self.var.w1[bioarea] + infiltration[bioarea]
        self.var.w2[bioarea] = self.var.w2[bioarea] + np.where(
            self.var.w1[bioarea] > self.var.ws1[bioarea],
            self.var.w1[bioarea] - self.var.ws1[bioarea],
            0,
        )  # now w2 could be over-saturated
        self.var.w1[bioarea] = np.minimum(self.var.ws1[bioarea], self.var.w1[bioarea])

        del infiltration
        assert (self.var.w1 >= 0).all()
        assert (self.var.w2 >= 0).all()
        assert (self.var.w3 >= 0).all()

        # Available water in both soil layers [m]
        availWater1 = np.maximum(0.0, self.var.w1[bioarea] - self.var.wres1[bioarea])
        availWater2 = np.maximum(0.0, self.var.w2[bioarea] - self.var.wres2[bioarea])
        availWater3 = np.maximum(0.0, self.var.w3[bioarea] - self.var.wres3[bioarea])

        satTerm2 = availWater2 / (self.var.ws2[bioarea] - self.var.wres2[bioarea])
        satTerm3 = availWater3 / (self.var.ws3[bioarea] - self.var.wres3[bioarea])

        satTerm2[satTerm2 < 0] = 0
        satTerm2[satTerm2 > 1] = 1
        satTerm3[satTerm3 < 0] = 0
        satTerm3[satTerm3 > 1] = 1

        # Saturation term in Van Genuchten equation (always between 0 and 1)
        assert (satTerm2 >= 0).all() and (satTerm2 <= 1).all()
        assert (satTerm3 >= 0).all() and (satTerm3 <= 1).all()

        kUnSat2 = (
            self.var.KSat2[bioarea]
            * np.sqrt(satTerm2)
            * np.square(
                1
                - (
                    1
                    - satTerm2
                    ** (
                        1
                        / (self.var.lambda2[bioarea] / (self.var.lambda2[bioarea] + 1))
                    )
                )
                ** (self.var.lambda2[bioarea] / (self.var.lambda2[bioarea] + 1))
            )
        )
        kUnSat3 = (
            self.var.KSat3[bioarea]
            * np.sqrt(satTerm3)
            * np.square(
                1
                - (
                    1
                    - satTerm3
                    ** (
                        1
                        / (self.var.lambda3[bioarea] / (self.var.lambda3[bioarea] + 1))
                    )
                )
                ** (self.var.lambda3[bioarea] / (self.var.lambda3[bioarea] + 1))
            )
        )

        ## ----------------------------------------------------------
        # Capillar Rise
        satTermFC1 = np.maximum(0.0, self.var.w1[bioarea] - self.var.wres1[bioarea]) / (
            self.var.wfc1[bioarea] - self.var.wres1[bioarea]
        )
        satTermFC2 = np.maximum(0.0, self.var.w2[bioarea] - self.var.wres2[bioarea]) / (
            self.var.wfc2[bioarea] - self.var.wres2[bioarea]
        )

        capRise1 = np.minimum(
            np.maximum(0.0, (1 - satTermFC1) * kUnSat2), self.var.kunSatFC12[bioarea]
        )
        capRise2 = np.minimum(
            np.maximum(0.0, (1 - satTermFC2) * kUnSat3), self.var.kunSatFC23[bioarea]
        )

        capRise2 = np.minimum(capRise2, availWater3)

        del satTermFC1
        del satTermFC2

        assert (self.var.w1 >= 0).all()
        assert (self.var.w2 >= 0).all()
        assert (self.var.w3 >= 0).all()

        self.var.w1[bioarea] = self.var.w1[bioarea] + capRise1
        self.var.w2[bioarea] = self.var.w2[bioarea] - capRise1 + capRise2
        self.var.w3[bioarea] = self.var.w3[bioarea] - capRise2

        assert (self.var.w1 >= 0).all()
        assert (self.var.w2 >= 0).all()
        assert (self.var.w3 >= 0).all()

        del capRise1
        del capRise2

        # Percolation -----------------------------------------------
        # Available water in both soil layers [m]
        availWater1 = np.maximum(0.0, self.var.w1[bioarea] - self.var.wres1[bioarea])
        availWater2 = np.maximum(0.0, self.var.w2[bioarea] - self.var.wres2[bioarea])
        availWater3 = np.maximum(0.0, self.var.w3[bioarea] - self.var.wres3[bioarea])

        # Available storage capacity in subsoil
        capLayer2 = self.var.ws2[bioarea] - self.var.w2[bioarea]
        capLayer3 = self.var.ws3[bioarea] - self.var.w3[bioarea]

        satTerm1 = availWater1 / (self.var.ws1[bioarea] - self.var.wres1[bioarea])
        satTerm2 = availWater2 / (self.var.ws2[bioarea] - self.var.wres2[bioarea])
        satTerm3 = availWater3 / (self.var.ws3[bioarea] - self.var.wres3[bioarea])

        # Saturation term in Van Genuchten equation (always between 0 and 1)
        satTerm1 = np.maximum(np.minimum(satTerm1, 1.0), 0)
        satTerm2 = np.maximum(np.minimum(satTerm2, 1.0), 0)
        satTerm3 = np.maximum(np.minimum(satTerm3, 1.0), 0)

        # Unsaturated conductivity
        kUnSat1 = (
            self.var.KSat1[bioarea]
            * np.sqrt(satTerm1)
            * np.square(
                1
                - (
                    1
                    - satTerm1
                    ** (
                        1
                        / (self.var.lambda1[bioarea] / (self.var.lambda1[bioarea] + 1))
                    )
                )
                ** (self.var.lambda1[bioarea] / (self.var.lambda1[bioarea] + 1))
            )
        )
        kUnSat2 = (
            self.var.KSat2[bioarea]
            * np.sqrt(satTerm2)
            * np.square(
                1
                - (
                    1
                    - satTerm2
                    ** (
                        1
                        / (self.var.lambda2[bioarea] / (self.var.lambda2[bioarea] + 1))
                    )
                )
                ** (self.var.lambda2[bioarea] / (self.var.lambda2[bioarea] + 1))
            )
        )
        kUnSat3 = (
            self.var.KSat3[bioarea]
            * np.sqrt(satTerm3)
            * np.square(
                1
                - (
                    1
                    - satTerm3
                    ** (
                        1
                        / (self.var.lambda3[bioarea] / (self.var.lambda3[bioarea] + 1))
                    )
                )
                ** (self.var.lambda3[bioarea] / (self.var.lambda3[bioarea] + 1))
            )
        )

        self.model.NoSubSteps = 3
        DtSub = 1.0 / self.model.NoSubSteps

        # Copy current value of W1 and W2 to temporary variables,
        # because computed fluxes may need correction for storage
        # capacity of subsoil and in case soil is frozen (after loop)
        wtemp1 = self.var.w1[bioarea].copy()
        wtemp2 = self.var.w2[bioarea].copy()
        wtemp3 = self.var.w3[bioarea].copy()

        # Initialize top- to subsoil flux (accumulated value for all sub-steps)
        # Initialize fluxes out of subsoil (accumulated value for all sub-steps)
        perc1to2 = self.var.zeros(bioarea.size, dtype=np.float32)
        perc2to3 = self.var.zeros(bioarea.size, dtype=np.float32)
        perc3toGW = self.var.full_compressed(0, dtype=np.float32)

        assert (self.var.w1 >= 0).all()
        assert (self.var.w2 >= 0).all()
        assert (self.var.w3 >= 0).all()

        # Start iterating

        for i in range(self.model.NoSubSteps):
            if i > 0:
                # Saturation term in Van Genuchten equation
                satTerm1 = np.maximum(0.0, wtemp1 - self.var.wres1[bioarea]) / (
                    self.var.ws1[bioarea] - self.var.wres1[bioarea]
                )
                satTerm2 = np.maximum(0.0, wtemp2 - self.var.wres2[bioarea]) / (
                    self.var.ws2[bioarea] - self.var.wres2[bioarea]
                )
                satTerm3 = np.maximum(0.0, wtemp3 - self.var.wres3[bioarea]) / (
                    self.var.ws3[bioarea] - self.var.wres3[bioarea]
                )

                satTerm1 = np.maximum(np.minimum(satTerm1, 1.0), 0)
                satTerm2 = np.maximum(np.minimum(satTerm2, 1.0), 0)
                satTerm3 = np.maximum(np.minimum(satTerm3, 1.0), 0)

                # Unsaturated hydraulic conductivities
                kUnSat1 = (
                    self.var.KSat1[bioarea]
                    * np.sqrt(satTerm1)
                    * np.square(
                        1
                        - (
                            1
                            - satTerm1
                            ** (
                                1
                                / (
                                    self.var.lambda1[bioarea]
                                    / (self.var.lambda1[bioarea] + 1)
                                )
                            )
                        )
                        ** (self.var.lambda1[bioarea] / (self.var.lambda1[bioarea] + 1))
                    )
                )
                kUnSat2 = (
                    self.var.KSat2[bioarea]
                    * np.sqrt(satTerm2)
                    * np.square(
                        1
                        - (
                            1
                            - satTerm2
                            ** (
                                1
                                / (
                                    self.var.lambda2[bioarea]
                                    / (self.var.lambda2[bioarea] + 1)
                                )
                            )
                        )
                        ** (self.var.lambda2[bioarea] / (self.var.lambda2[bioarea] + 1))
                    )
                )
                kUnSat3 = (
                    self.var.KSat3[bioarea]
                    * np.sqrt(satTerm3)
                    * np.square(
                        1
                        - (
                            1
                            - satTerm3
                            ** (
                                1
                                / (
                                    self.var.lambda3[bioarea]
                                    / (self.var.lambda3[bioarea] + 1)
                                )
                            )
                        )
                        ** (self.var.lambda3[bioarea] / (self.var.lambda3[bioarea] + 1))
                    )
                )

            # Flux from top- to subsoil
            subperc1to2 = np.minimum(
                availWater1, np.minimum(kUnSat1 * DtSub, capLayer2)
            )
            subperc2to3 = np.minimum(
                availWater2, np.minimum(kUnSat2 * DtSub, capLayer3)
            )
            subperc3toGW = np.minimum(
                availWater3, np.minimum(kUnSat3 * DtSub, availWater3)
            ) * (1 - self.var.capriseindex[bioarea])

            # When the soil is frozen (frostindex larger than threshold), no perc1 and 2
            subperc1to2 = np.where(
                self.var.FrostIndex[bioarea] > self.var.FrostIndexThreshold,
                0,
                subperc1to2,
            )
            subperc2to3 = np.where(
                self.var.FrostIndex[bioarea] > self.var.FrostIndexThreshold,
                0,
                subperc2to3,
            )

            # Update water balance for all layers
            availWater1 = availWater1 - subperc1to2
            availWater2 = availWater2 + subperc1to2 - subperc2to3
            availWater3 = availWater3 + subperc2to3 - subperc3toGW
            # Update WTemp1 and WTemp2

            wtemp1 = availWater1 + self.var.wres1[bioarea]
            wtemp2 = availWater2 + self.var.wres2[bioarea]
            wtemp3 = availWater3 + self.var.wres3[bioarea]

            # Update available storage capacity in layer 2,3
            capLayer2 = self.var.ws2[bioarea] - wtemp2
            capLayer3 = self.var.ws3[bioarea] - wtemp3

            perc1to2 += subperc1to2
            perc2to3 += subperc2to3
            perc3toGW[bioarea] += subperc3toGW

            assert not np.isnan(perc1to2).any()
            assert not np.isnan(perc2to3).any()
            assert not np.isnan(perc3toGW[bioarea]).any()

            del subperc1to2
            del subperc2to3
            del subperc3toGW

            del kUnSat1
            del kUnSat2
            del kUnSat3

        del satTerm1
        del satTerm2
        del satTerm3

        del capLayer2
        del capLayer3

        del wtemp1
        del wtemp2
        del wtemp3

        del availWater1
        del availWater2
        del availWater3

        # Update soil moisture
        assert (self.var.w1 >= 0).all()
        self.var.w1[bioarea] = self.var.w1[bioarea] - perc1to2
        assert (self.var.w1 >= 0).all()
        self.var.w2[bioarea] = self.var.w2[bioarea] + perc1to2 - perc2to3
        assert (self.var.w2 >= 0).all()
        self.var.w3[bioarea] = self.var.w3[bioarea] + perc2to3 - perc3toGW[bioarea]
        assert (self.var.w3 >= 0).all()

        assert not np.isnan(self.var.w1).any()
        assert not np.isnan(self.var.w2).any()
        assert not np.isnan(self.var.w3).any()

        del perc1to2
        del perc2to3

        # self.var.theta1[bioarea] = self.var.w1[bioarea] / rootDepth1[bioarea]
        # self.var.theta2[bioarea] = self.var.w2[bioarea] / rootDepth2[bioarea]
        # self.var.theta3[bioarea] = self.var.w3[bioarea] / rootDepth3[bioarea]

        # ---------------------------------------------------------------------------------------------
        # Calculate interflow

        # total actual transpiration
        # self.var.actTransTotal[No] = actTrans[0] + actTrans[1] + actTrans[2]
        # self.var.actTransTotal[No] =  np.sum(actTrans, axis=0)

        # This relates to deficit conditions, and calculating the ratio of actual to potential transpiration

        # total actual evaporation + transpiration
        self.var.actualET[bioarea] = (
            self.var.actualET[bioarea]
            + self.var.actBareSoilEvap[bioarea]
            + openWaterEvap[bioarea]
            + self.var.actTransTotal[bioarea]
        )

        #  actual evapotranspiration can be bigger than pot, because openWater is taken from pot open water evaporation, therefore self.var.totalPotET[No] is adjusted
        # totalPotET[bioarea] = np.maximum(totalPotET[bioarea], self.var.actualET[bioarea])

        # net percolation between upperSoilStores (positive indicating downward direction)
        # elf.var.netPerc[No] = perc[0] - capRise[0]
        # self.var.netPercUpper[No] = perc[1] - capRise[1]

        # groundwater recharge
        toGWorInterflow = perc3toGW[bioarea] + prefFlow[bioarea]

        interflow = self.var.full_compressed(0, dtype=np.float32)
        interflow[bioarea] = self.var.percolationImp[bioarea] * toGWorInterflow

        groundwater_recharge = self.var.full_compressed(0, dtype=np.float32)
        groundwater_recharge[bioarea] = (
            1 - self.var.percolationImp[bioarea]
        ) * toGWorInterflow

        assert not np.isnan(interflow).any()
        assert not np.isnan(groundwater_recharge).any()

        if checkOption("calcWaterBalance"):
            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[
                    self.var.natural_available_water_infiltration[bioarea],
                    capillar[bioarea],
                    self.var.actual_irrigation_consumption[bioarea],
                ],
                outfluxes=[
                    directRunoff[bioarea],
                    perc3toGW[bioarea],
                    prefFlow[bioarea],
                    self.var.actTransTotal[bioarea],
                    self.var.actBareSoilEvap[bioarea],
                    openWaterEvap[bioarea],
                ],
                prestorages=[
                    w1_pre[bioarea],
                    w2_pre[bioarea],
                    w3_pre[bioarea],
                    topwater_pre[bioarea],
                ],
                poststorages=[
                    self.var.w1[bioarea],
                    self.var.w2[bioarea],
                    self.var.w3[bioarea],
                    self.var.topwater[bioarea],
                ],
                tollerance=1e-6,
            )

            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[
                    self.var.natural_available_water_infiltration[bioarea],
                    capillar[bioarea],
                    self.var.actual_irrigation_consumption[bioarea],
                    self.var.snowEvap[bioarea],
                    self.var.interceptEvap[bioarea],
                ],
                outfluxes=[
                    directRunoff[bioarea],
                    interflow[bioarea],
                    groundwater_recharge[bioarea],
                    self.var.actualET[bioarea],
                ],
                prestorages=[
                    w1_pre[bioarea],
                    w2_pre[bioarea],
                    w3_pre[bioarea],
                    topwater_pre[bioarea],
                ],
                poststorages=[
                    self.var.w1[bioarea],
                    self.var.w2[bioarea],
                    self.var.w3[bioarea],
                    self.var.topwater[bioarea],
                ],
                tollerance=1e-6,
            )

        # print(time() - t0)

        return (
            interflow,
            directRunoff,
            groundwater_recharge,
            perc3toGW,
            prefFlow,
            openWaterEvap,
        )
