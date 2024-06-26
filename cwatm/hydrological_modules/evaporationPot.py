# -------------------------------------------------------------------------
# Name:        POTENTIAL REFERENCE EVAPO(TRANSPI)RATION
# Purpose:
#
# Author:      PB
#
# Created:     10/01/2017
# Copyright:   (c) PB 2017
# -------------------------------------------------------------------------

import numpy as np
from cwatm.management_modules.data_handling import (
    loadmap,
    loadmap,
    cbinding,
)

from geb.workflows import TimingModule


class evaporationPot(object):
    """
    POTENTIAL REFERENCE EVAPO(TRANSPI)RATION
    Calculate potential evapotranspiration from climate data mainly based on FAO 56 and LISVAP
    Based on Penman Monteith

    References:
        http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation
        http://www.fao.org/docrep/X0490E/x0490e06.htm  http://www.fao.org/docrep/X0490E/x0490e06.htm
        https://ec.europa.eu/jrc/en/publication/eur-scientific-and-technical-research-reports/lisvap-evaporation-pre-processor-lisflood-water-balance-and-flood-simulation-model

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    crop_factor_calibration_factor           calibrated factor of crop KC factor                                               --
    pet_modus             Flag: index which ETP approach is used e.g. 1 for Penman-Monteith                 --
    AlbedoCanopy          Albedo of vegetation canopy (FAO,1998) default = 0.23                              --
    AlbedoSoil            Albedo of bare soil surface (Supit et. al. 1994) default = 0.15                   --
    AlbedoWater           Albedo of water surface (Supit et. al. 1994) default = 0.05                       --
    co2
    TMin                  minimum air temperature                                                           K
    TMax                  maximum air temperature                                                           K
    Psurf                 Instantaneous surface pressure                                                    Pa
    Qair                  specific humidity                                                                 kg/kg
    Tavg                  average air Temperature (input for the model)                                     K
    Rsdl                  long wave downward surface radiation fluxes                                       W/m2
    albedoLand            albedo from land surface (from GlobAlbedo database)                               --
    albedoOpenWater       albedo from open water surface (from GlobAlbedo database)                         --
    Rsds                  short wave downward surface radiation fluxes                                      W/m2
    Wind                  wind speed                                                                        m/s
    ETRef                 potential evapotranspiration rate from reference crop                             m
    EWRef                 potential evaporation rate from water surface                                     m
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        """
        The constructor evaporationPot
        """
        self.var = model.data.HRU
        self.model = model

    def initial(self):
        """
        Initial part of evaporation type module
        Load inictial parameters

        """
        self.var.crop_factor_calibration_factor = cbinding(
            "crop_factor_calibration_factor"
        )

        self.var.crop_factor_calibration_factor = self.model.data.to_HRU(
            data=self.var.crop_factor_calibration_factor, fn=None
        )

        self.var.AlbedoCanopy = loadmap("AlbedoCanopy")
        self.var.AlbedoSoil = loadmap("AlbedoSoil")
        self.var.AlbedoWater = loadmap("AlbedoWater")

    def dynamic(self):
        """
        Dynamic part of the potential evaporation module
        Based on Penman Monteith - FAO 56

        """

        timer = TimingModule("Potential evaporation")

        tas_C = self.var.tas - 273.15
        tasmin_C = self.var.tasmin - 273.15
        tasmax_C = self.var.tasmax - 273.15

        timer.new_split("Read data")

        # http://www.fao.org/docrep/X0490E/x0490e07.htm   equation 11/12
        ESatmin = 0.6108 * np.exp((17.27 * tasmin_C) / (tasmin_C + 237.3))
        ESatmax = 0.6108 * np.exp((17.27 * tasmax_C) / (tasmax_C + 237.3))
        saturated_vapour_pressure = (ESatmin + ESatmax) / 2.0  # [KPa]

        timer.new_split("Saturation vapour pressure")

        # Up longwave radiation [MJ/m2/day]
        rlus_MJ_m2_day = (
            4.903e-9 * (((tasmin_C + 273.16) ** 4) + ((tasmax_C + 273.16) ** 4)) / 2
        )  # rlus = Surface Upwelling Longwave Radiation

        ps_kPa = self.var.ps * 0.001
        psychrometric_constant = 0.665e-3 * ps_kPa
        # psychrometric constant [kPa C-1]
        # http://www.fao.org/docrep/X0490E/x0490e07.htm  Equation 8
        # see http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation

        timer.new_split("Psychrometric constant")

        # calculate vapor pressure
        # Fao 56 Page 36
        # calculate actual vapour pressure

        actual_vapour_pressure = saturated_vapour_pressure * self.var.hurs / 100.0
        # longwave radiation balance

        rlds_MJ_m2_day = self.var.rlds * 0.0864  # 86400 * 1E-6
        net_longwave_radation_MJ_m2_day = rlus_MJ_m2_day - rlds_MJ_m2_day

        # ************************************************************
        # ***** NET ABSORBED RADIATION *******************************
        # ************************************************************

        rsds_MJ_m2_day = self.var.rsds * 0.0864  # 86400 * 1E-6
        # net absorbed radiation of reference vegetation canopy [mm/d]
        RNA = np.maximum(
            (1 - self.var.AlbedoCanopy) * rsds_MJ_m2_day
            - net_longwave_radation_MJ_m2_day,
            0.0,
        )
        # net absorbed radiation of bare soil surface
        RNAWater = np.maximum(
            (1 - self.var.AlbedoWater) * rsds_MJ_m2_day
            - net_longwave_radation_MJ_m2_day,
            0.0,
        )
        # net absorbed radiation of water surface

        timer.new_split("Radiation")

        vapour_pressure_deficit = np.maximum(
            saturated_vapour_pressure - actual_vapour_pressure, 0.0
        )
        slope_of_saturated_vapour_pressure_curve = (
            4098.0 * saturated_vapour_pressure
        ) / ((tas_C + 237.3) ** 2)
        # slope of saturated vapour pressure curve [kPa/deg C]
        # Equation 13 Chapter 3

        timer.new_split("Vapour pressure")

        # Chapter 2 Equation 6
        # Adjust wind speed for measurement height: wind speed measured at
        # 10 m, but needed at 2 m height
        # Shuttleworth, W.J. (1993) in Maidment, D.R. (1993), p. 4.36
        wind_2m = self.var.sfcWind * 0.749

        # TODO: update this properly following PCR-GLOBWB (https://github.com/UU-Hydro/PCR-GLOBWB_model/blob/0511485ad3ac0a1367d9d4918d2f61ae0fa0e900/model/evaporation/ref_pot_et_penman_monteith.py#L227)

        denominator = (
            slope_of_saturated_vapour_pressure_curve
            + psychrometric_constant * (1 + 0.34 * wind_2m)
        )

        # TODO: check if this is correct. Specifically the replacement 0.408 constant with 1 / LatHeatVap. In the original code
        # it seems that the latent heat is only applied to the first nominator and not to the second one, see: https://en.wikipedia.org/wiki/Penman%E2%80%93Monteith_equation

        # latent heat of vaporization [MJ/kg]
        LatHeatVap = 2.501 - 0.002361 * tas_C
        # the 0.408 constant is replace by 1/LatHeatVap
        RNAN = RNA / LatHeatVap * slope_of_saturated_vapour_pressure_curve / denominator
        RNANWater = (
            RNAWater
            / LatHeatVap
            * slope_of_saturated_vapour_pressure_curve
            / denominator
        )

        EA = (
            psychrometric_constant
            * 900
            / (tas_C + 273.16)
            * wind_2m
            * vapour_pressure_deficit
            / denominator
        )

        timer.new_split("Penman-Monteith")

        # Potential evapo(transpi)ration is calculated for two reference surfaces:
        # 1. Reference vegetation canopy (ETRef)
        # 2. Open water surface (EWRef)

        self.var.ETRef = (RNAN + EA) * 0.001
        # potential reference evapotranspiration rate [m/day]  # from mm to m with 0.001
        # potential evaporation rate from a bare soil surface [m/day]

        self.var.EWRef = (RNANWater + EA) * 0.001
        # potential evaporation rate from water surface [m/day]

        # -> here we are at ET0 (see http://www.fao.org/docrep/X0490E/x0490e04.htm#TopOfPage figure 4:)

        timer.new_split("ETRef and EWRef")

        if self.model.timing:
            print(timer)

        self.model.agents.crop_farmers.save_water_deficit()
