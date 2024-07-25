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
import rasterio
from cwatm.management_modules.data_handling import loadmap, checkOption
from pathlib import Path
from geb.workflows import TimingModule


def get_critical_soil_moisture_content(p, wfc, wwp):
    """
    "The critical soil moisture content is defined as the quantity of stored soil moisture below
    which water uptake is impaired and the crop begins to close its stomata. It is not a fixed
    value. Restriction of water uptake due to water stress starts at a higher water content
    when the potential transpiration rate is higher" (Van Diepen et al., 1988: WOFOST 6.0, p.86)

    A higher p value means that the critical soil moisture content is higher, i.e. the plant can
    extract water from the soil at a lower soil moisture content. Thus when p is 1 the critical
    soil moisture content is equal to the wilting point, and when p is 0 the critical soil moisture
    content is equal to the field capacity.
    """
    return (1 - p) * (wfc - wwp) + wwp


def get_aeration_stress_threshold(
    cum_soil_layer_height, soil_layer_height, root_depth, ws, crop_aer
):
    if cum_soil_layer_height > root_depth:
        factor = 1 - ((cum_soil_layer_height - root_depth) / soil_layer_height)
    else:
        factor = 1

    # Water storage in root zone at aeration stress threshold (mm)
    WrAer = round(factor * (ws - (crop_aer / 100)) * soil_layer_height, 2)

    return WrAer


def sum_aeration_stress_threshold(aeration_stress_threshold_per_layer, root_depth):
    # Root zone water content at aeration stress threshold (m3/m3)
    return aeration_stress_threshold_per_layer.sum(axis=0) / root_depth


def aeration_stress_factor(ws, w, waer, aer_days, crop_lag_aer):
    if w > waer:
        # Calculate aeration stress coefficient
        if aer_days < crop_lag_aer:
            stress = 1 - ((ws - w) / (ws - waer))
            ksa_aer = 1 - ((aer_days / 3) * stress)
        elif aer_days >= crop_lag_aer:
            ksa_aer = (ws - w) / (ws - waer)

        # Increment aeration days counter
        aer_days = aer_days + 1
        if aer_days > crop_lag_aer:
            aer_days = crop_lag_aer

    else:
        # Set aeration stress coefficient to one (no stress value)
        ksa_aer = 1
        # Reset aeration days counter
        aer_days = 0

    return ksa_aer, aer_days


def get_available_water(w, wwp):
    return np.maximum(0.0, w - wwp)


def get_maximum_water_content(wfc, wwp):
    return wfc - wwp


def get_fraction_easily_available_soil_water(
    crop_group_number, potential_evapotranspiration
):
    """
    Calculate the fraction of easily available soil water, based on crop group number and potential evapotranspiration
    following Van Diepen et al., 1988: WOFOST 6.0, p.87

    Parameters
    ----------
    crop_group_number : np.ndarray
        The crop group number is a indicator of adaptation to dry climate,
        Van Diepen et al., 1988: WOFOST 6.0, p.87
    potential_evapotranspiration : np.ndarray
        Potential evapotranspiration in m

    Returns
    -------
    np.ndarray
        The fraction of easily available soil water, p is closer to 0 if evapo is bigger and cropgroup is smaller
    """
    potential_evapotranspiration_cm = potential_evapotranspiration * 100.0

    p = 1 / (0.76 + 1.5 * potential_evapotranspiration_cm) - 0.10 * (
        5 - crop_group_number
    )

    # Additional correction for crop groups 1 and 2
    p = np.where(
        crop_group_number <= 2.5,
        p
        + (potential_evapotranspiration_cm - 0.6)
        / (crop_group_number * (crop_group_number + 3)),
        p,
    )
    assert not (np.isnan(p)).any()
    return np.clip(p, 0, 1)


def get_critical_water_level(p, wfc, wwp):
    return np.maximum(get_critical_soil_moisture_content(p, wfc, wwp) - wwp, 0)


def get_total_transpiration_reduction_factor(
    transpiration_reduction_factor_per_layer, root_ratios, soil_layer_height
):
    transpiration_reduction_factor_relative_contribution_per_layer = (
        soil_layer_height * root_ratios
    )
    transpiration_reduction_factor_total = np.sum(
        transpiration_reduction_factor_relative_contribution_per_layer
        / transpiration_reduction_factor_relative_contribution_per_layer.sum(axis=0)
        * transpiration_reduction_factor_per_layer,
        axis=0,
    )
    return transpiration_reduction_factor_total


def get_transpiration_reduction_factor(w, wwp, wcrit):
    factor = np.clip((w - wwp) / (wcrit - wwp), 0, 1)
    return np.nan_to_num(factor)


def get_root_ratios(
    root_depth,
    soil_layer_height,
):
    remaining_root_depth = root_depth.copy()
    root_ratios = np.zeros_like(soil_layer_height)
    for layer in range(soil_layer_height.shape[0]):
        mask = remaining_root_depth > 0
        root_ratios[layer, mask] = np.minimum(
            remaining_root_depth[mask] / soil_layer_height[layer, mask], 1
        )
        remaining_root_depth[mask] -= soil_layer_height[layer, mask]
    return root_ratios


def get_crop_group_number(
    crop_map, crop_group_numbers, land_use_type, natural_crop_groups
):
    crop_group_map = np.take(crop_group_numbers, crop_map)
    crop_group_map[crop_map == -1] = np.nan

    natural_land = np.isin(land_use_type, (0, 1))
    crop_group_map[natural_land] = natural_crop_groups[natural_land]
    return crop_group_map


def get_unsaturated_hydraulic_conductivity(
    w, wres, ws, lambda_, saturated_hydraulic_conductivity
):
    saturation_term = np.clip((w - wres) / (ws - wres), 0, 1)
    residual_moisture = lambda_ / (lambda_ + 1)

    return (
        saturated_hydraulic_conductivity
        * saturation_term**0.5
        * (1 - (1 - saturation_term ** (1 / residual_moisture)) ** residual_moisture)
        ** 2
    )


class soil(object):
    """
    **SOIL**


    Calculation vertical transfer of water based on Arno scheme


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    storGroundwater       simulated groundwater storage                                                     m
    capRiseFrac           fraction of a grid cell where capillar rise may happen                            m
    cropKC                crop coefficient for each of the 4 different land cover types (forest, irrigated  --
    EWRef                 potential evaporation rate from water surface                                     m
    capillar              Simulated flow from groundwater to the third CWATM soil layer                     m
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m
    potTranspiration      Potential transpiration (after removing of evaporation)                           m
    actualET              simulated evapotranspiration from soil, flooded area and vegetation               m
    soilLayers            Number of soil layers                                                             --
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
        # soil water depletion fraction, Van Diepen et al., 1988: WOFOST 6.0, p.86, Doorenbos et. al 1978
        # crop groups for formular in van Diepen et al, 1988

        with rasterio.open(self.model.model_structure["grid"]["soil/cropgrp"]) as src:
            natural_crop_groups = self.model.data.grid.compress(src.read(1))
            self.var.natural_crop_groups = self.model.data.to_HRU(
                data=natural_crop_groups
            )

        # ------------ Preferential Flow constant ------------------------------------------
        self.var.cPrefFlow = self.model.data.to_HRU(
            data=loadmap("preferentialFlowConstant"), fn=None
        )

        def create_ini(yaml, idx, plantFATE_cluster, biodiversity_scenario):
            out_dir = self.model.simulation_root / "plantFATE" / f"cell_{idx}"
            out_dir.mkdir(parents=True, exist_ok=True)
            ini_file = out_dir / f"p_daily.ini"

            yaml["> STRINGS"]["outDir"] = out_dir
            if self.model.spinup is True:
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

        timer = TimingModule("Soil")
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
        # CAPRISE from GW to soilayer 1 , if this is full it is send to saverunoffromGW
        saverunoffromGW = np.where(
            self.var.w1[bioarea] > self.var.ws1[bioarea],
            self.var.w1[bioarea] - self.var.ws1[bioarea],
            0,
        )
        self.var.w1[bioarea] = np.minimum(self.var.ws1[bioarea], self.var.w1[bioarea])

        assert (self.var.w1 >= 0).all()
        assert (self.var.w2 >= 0).all()
        assert (self.var.w3 >= 0).all()

        timer.new_split("Various")
        # ---------------------------------------------------------
        # calculate transpiration
        # ***** SOIL WATER STRESS ************************************

        # load crop group number
        crop_group_number = get_crop_group_number(
            self.var.crop_map,
            self.model.agents.crop_farmers.crop_data["crop_group_number"].values,
            self.var.land_use_type,
            self.var.natural_crop_groups,
        )

        # p is between 0 and 1 => if p =1 wcrit = wwp, if p= 0 wcrit = wfc
        p = get_fraction_easily_available_soil_water(
            crop_group_number[bioarea], totalPotET[bioarea]
        )

        root_ratios = get_root_ratios(
            self.var.root_depth[bioarea],
            self.var.soil_layer_height[:, bioarea],
        )

        critical_soil_moisture1 = get_critical_soil_moisture_content(
            p, self.var.wfc1[bioarea], self.var.wwp1[bioarea]
        )
        critical_soil_moisture2 = get_critical_soil_moisture_content(
            p, self.var.wfc2[bioarea], self.var.wwp2[bioarea]
        )
        critical_soil_moisture3 = get_critical_soil_moisture_content(
            p, self.var.wfc3[bioarea], self.var.wwp3[bioarea]
        )

        # del p

        transpiration_reduction_factor_per_layer = np.zeros_like(root_ratios)

        transpiration_reduction_factor_per_layer[0] = (
            get_transpiration_reduction_factor(
                self.var.w1[bioarea],
                self.var.wwp1[bioarea],
                critical_soil_moisture1,
            )
        )
        transpiration_reduction_factor_per_layer[1] = (
            get_transpiration_reduction_factor(
                self.var.w2[bioarea],
                self.var.wwp2[bioarea],
                critical_soil_moisture2,
            )
        )
        transpiration_reduction_factor_per_layer[2] = (
            get_transpiration_reduction_factor(
                self.var.w3[bioarea],
                self.var.wwp3[bioarea],
                critical_soil_moisture3,
            )
        )

        transpiration_reduction_factor_total = get_total_transpiration_reduction_factor(
            transpiration_reduction_factor_per_layer,
            root_ratios,
            self.var.soil_layer_height[:, bioarea],
        )

        # del critical_soil_moisture1
        # del critical_soil_moisture2
        # del critical_soil_moisture3

        TaMax = potTranspiration[bioarea] * transpiration_reduction_factor_total

        del potTranspiration

        # set transpiration to 0 when soil is frozen
        TaMax[self.var.FrostIndex[bioarea] > self.var.FrostIndexThreshold] = 0.0

        # self.model.data.grid.vapour_pressure_deficit = (
        #     self.calculate_vapour_pressure_deficit_kPa(
        #         self.model.data.grid.tas, self.model.data.grid.hurs
        #     )
        # )

        # self.model.data.grid.photosynthetic_photon_flux_density = (
        #     self.calculate_photosynthetic_photon_flux_density(self.model.data.grid.rsds)
        # )

        # soil_water_potential = self.calculate_soil_water_potential_MPa(
        #     self.var.w1 + self.var.w2 + self.var.w3,  # [m]
        #     self.var.wwp1 + self.var.wwp2 + self.var.wwp3,  # [m]
        #     self.var.wfc1 + self.var.wfc2 + self.var.wfc3,  # [m]
        #     self.var.soil_layer_height[0]
        #     + self.var.soil_layer_height[1]
        #     + self.var.soil_layer_height[2],  # [m]
        #     wilting_point=-1500,  # kPa
        #     field_capacity=-33,  # kPa
        # )
        # soil_water_potential_plantFATE_HRUs = np.where(
        #     self.var.land_use_type == 0,
        #     soil_water_potential,
        #     np.nan,
        # )
        # self.model.data.grid.soil_water_potential = self.model.data.to_grid(
        #     HRU_data=soil_water_potential_plantFATE_HRUs, fn="weightednanmean"
        # )

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

                    if self.model.current_timestep == 1:
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

            print("ADJUST FOR ROOT DEPTH")

            ta1 = np.maximum(
                np.minimum(
                    TaMax,  # * self.var.adjRoot[0][bioarea],
                    self.var.w1[bioarea] - self.var.wwp1[bioarea],
                ),
                0.0,
            )
            ta2 = np.maximum(
                np.minimum(
                    TaMax,  # * self.var.adjRoot[1][bioarea],
                    self.var.w2[bioarea] - self.var.wwp2[bioarea],
                ),
                0.0,
            )
            ta3 = np.maximum(
                np.minimum(
                    TaMax,  # * self.var.adjRoot[2][bioarea],
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
            root_distribution_per_layer_non_normalized = (
                self.var.soil_layer_height[:, bioarea] * root_ratios
            )

            root_distribution_per_layer_rws_corrected_non_normalized = (
                root_distribution_per_layer_non_normalized
                * transpiration_reduction_factor_per_layer
            )
            root_distribution_per_layer_rws_corrected = (
                root_distribution_per_layer_rws_corrected_non_normalized
                / root_distribution_per_layer_rws_corrected_non_normalized.sum(axis=0)
            )
            # when no water is available, no transpiration can occur. Avoid nan values
            # by setting the transpiration to 0
            root_distribution_per_layer_rws_corrected[
                :,
                root_distribution_per_layer_rws_corrected_non_normalized.sum(axis=0)
                == 0,
            ] = 0

            ta = TaMax * root_distribution_per_layer_rws_corrected
            ta[:, (TaMax == 0)] = 0

        del TaMax

        self.var.w1[bioarea] = self.var.w1[bioarea] - ta[0]
        self.var.w2[bioarea] = self.var.w2[bioarea] - ta[1]
        self.var.w3[bioarea] = self.var.w3[bioarea] - ta[2]

        assert (self.var.w1 >= 0).all()
        assert (self.var.w2 >= 0).all()
        assert (self.var.w3 >= 0).all()

        self.var.actTransTotal[bioarea] = ta.sum(axis=0)

        del ta

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

        timer.new_split("Water stress")

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
            0.0,
            self.var.topwater[paddy_irrigated_land]
            - self.model.agents.crop_farmers.max_paddy_water_level,
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

        directRunoff[bioarea] = directRunoff[bioarea] + saverunoffromGW
        # ADDING EXCESS WATER FROM GW CAPILLARY RISE

        del saverunoffromGW

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

        timer.new_split("Infiltration")

        kUnSat2 = get_unsaturated_hydraulic_conductivity(
            self.var.w2[bioarea],
            self.var.wres2[bioarea],
            self.var.ws2[bioarea],
            self.var.lambda2[bioarea],
            self.var.KSat2[bioarea],
        )

        kUnSat3 = get_unsaturated_hydraulic_conductivity(
            self.var.w3[bioarea],
            self.var.wres3[bioarea],
            self.var.ws3[bioarea],
            self.var.lambda3[bioarea],
            self.var.KSat3[bioarea],
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

        availWater3 = np.maximum(0.0, self.var.w3[bioarea] - self.var.wres3[bioarea])
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

        NoSubSteps = 3
        DtSub = 1.0 / NoSubSteps

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

        timer.new_split("Fluxes - setup")

        for i in range(NoSubSteps):
            kUnSat1 = get_unsaturated_hydraulic_conductivity(
                wtemp1,
                self.var.wres1[bioarea],
                self.var.ws1[bioarea],
                self.var.lambda1[bioarea],
                self.var.KSat1[bioarea],
            )
            kUnSat2 = get_unsaturated_hydraulic_conductivity(
                wtemp2,
                self.var.wres2[bioarea],
                self.var.ws2[bioarea],
                self.var.lambda2[bioarea],
                self.var.KSat2[bioarea],
            )
            kUnSat3 = get_unsaturated_hydraulic_conductivity(
                wtemp3,
                self.var.wres3[bioarea],
                self.var.ws3[bioarea],
                self.var.lambda3[bioarea],
                self.var.KSat3[bioarea],
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

        # ---------------------------------------------------------------------------------------------
        # Calculate interflow

        # total actual transpiration
        # self.var.actTransTotal[No] = actTrans[0] + actTrans[1] + actTrans[2]
        # self.var.actTransTotal[No] =  np.sum(actTrans, axis=0)

        # This relates to deficit conditions, and calculating the ratio of actual to potential transpiration

        timer.new_split("Fluxes")

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

        timer.new_split("Finalizing")
        if self.model.timing:
            print(timer)

        return (
            interflow,
            directRunoff,
            groundwater_recharge,
            perc3toGW,
            prefFlow,
            openWaterEvap,
        )
