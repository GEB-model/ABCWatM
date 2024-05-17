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
from cwatm.management_modules.data_handling import loadmap, returnBool, loadmap
from cwatm.management_modules.globals import binding


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
    cropCorrect           calibrated factor of crop KC factor                                               --
    pet_modus             Flag: index which ETP approach is used e.g. 1 for Penman-Monteith                 --
    AlbedoCanopy          Albedo of vegetation canopy (FAO,1998) default =0.23                              --
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

        Note:
            Only run if *calc_evaporation* is True
        """

        # self.var.sumETRef = globals.inZero.copy()
        self.var.cropCorrect = loadmap("crop_correct")
        self.var.cropCorrect = self.model.data.to_HRU(
            data=self.var.cropCorrect, fn=None
        )

        self.var.AlbedoCanopy = loadmap("AlbedoCanopy")
        self.var.AlbedoSoil = loadmap("AlbedoSoil")
        self.var.AlbedoWater = loadmap("AlbedoWater")

    def dynamic(self):
        """
        Dynamic part of the potential evaporation module
        Based on Penman Monteith - FAO 56

        """
        tas_C = self.model.data.to_HRU(data=self.model.data.grid.tas, fn=None) - 273.15
        tasmin_C = (
            self.model.data.to_HRU(data=self.model.data.grid.tasmin, fn=None) - 273.15
        )
        tasmax_C = (
            self.model.data.to_HRU(data=self.model.data.grid.tasmax, fn=None) - 273.15
        )

        # http://www.fao.org/docrep/X0490E/x0490e07.htm   equation 11/12
        ESatmin = 0.6108 * np.exp((17.27 * tasmin_C) / (tasmin_C + 237.3))
        ESatmax = 0.6108 * np.exp((17.27 * tasmax_C) / (tasmax_C + 237.3))
        saturated_vapour_pressure = (ESatmin + ESatmax) / 2.0  # [KPa]

        # Up longwave radiation [MJ/m2/day]
        rlus_MJ_m2_day = (
            4.903e-9 * (((tasmin_C + 273.16) ** 4) + ((tasmax_C + 273.16) ** 4)) / 2
        )  # rlus = Surface Upwelling Longwave Radiation

        ps_kPa = self.model.data.to_HRU(data=self.model.data.grid.ps, fn=None) * 0.001
        psychrometric_constant = 0.665e-3 * ps_kPa
        # psychrometric constant [kPa C-1]
        # http://www.fao.org/docrep/X0490E/x0490e07.htm  Equation 8
        # see http://www.fao.org/docrep/X0490E/x0490e08.htm#penman%20monteith%20equation

        # calculate vapor pressure
        # Fao 56 Page 36
        # calculate actual vapour pressure
        if returnBool("useHuss"):
            raise NotImplementedError
        else:
            hurs = self.model.data.to_HRU(data=self.model.data.grid.hurs, fn=None)
            # if relative humidity
            actual_vapour_pressure = saturated_vapour_pressure * hurs / 100.0
            # longwave radiation balance

        rlds_MJ_m2_day = (
            self.model.data.to_HRU(data=self.model.data.grid.rlds, fn=None) * 0.0864
        )  # 86400 * 1E-6
        net_longwave_radation_MJ_m2_day = rlus_MJ_m2_day - rlds_MJ_m2_day

        # ************************************************************
        # ***** NET ABSORBED RADIATION *******************************
        # ************************************************************

        rsds_MJ_m2_day = (
            self.model.data.to_HRU(data=self.model.data.grid.rsds, fn=None) * 0.0864
        )  # 86400 * 1E-6
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

        vapour_pressure_deficit = np.maximum(
            saturated_vapour_pressure - actual_vapour_pressure, 0.0
        )
        slope_of_saturated_vapour_pressure_curve = (
            4098.0 * saturated_vapour_pressure
        ) / ((tas_C + 237.3) ** 2)
        # slope of saturated vapour pressure curve [kPa/deg C]
        # Equation 13 Chapter 3

        # Chapter 2 Equation 6
        # Adjust wind speed for measurement height: wind speed measured at
        # 10 m, but needed at 2 m height
        # Shuttleworth, W.J. (1993) in Maidment, D.R. (1993), p. 4.36
        wind_2m = (
            self.model.data.to_HRU(data=self.model.data.grid.sfcWind, fn=None) * 0.749
        )

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

        # Potential evapo(transpi)ration is calculated for two reference surfaces:
        # 1. Reference vegetation canopy (ETRef)
        # 2. Open water surface (EWRef)

        self.var.ETRef = (RNAN + EA) * 0.001
        # potential reference evapotranspiration rate [m/day]  # from mm to m with 0.001
        # potential evaporation rate from a bare soil surface [m/day]

        self.var.EWRef = (RNANWater + EA) * 0.001
        # potential evaporation rate from water surface [m/day]

        # -> here we are at ET0 (see http://www.fao.org/docrep/X0490E/x0490e04.htm#TopOfPage figure 4:)

        self.ETref_forest = self.var.full_compressed(np.nan, dtype=np.float32)
        self.ETref_agriculture = self.var.full_compressed(np.nan, dtype=np.float32)
        self.ETref_grassland = self.var.full_compressed(np.nan, dtype=np.float32)
        self.averagetemp_forest = self.var.full_compressed(np.nan, dtype=np.float32)
        self.averagetemp_agriculture = self.var.full_compressed(np.nan, dtype=np.float32)
        self.averagetemp_grassland = self.var.full_compressed(np.nan, dtype=np.float32)
        self.humidity_forest = self.var.full_compressed(np.nan, dtype=np.float32)
        self.humidity_agriculture = self.var.full_compressed(np.nan, dtype=np.float32)
        self.humidity_grassland = self.var.full_compressed(np.nan, dtype=np.float32)

        

        self.ETref_forest[:] = sum(self.var.ETRef[self.var.land_use_indices_forest] * self.model.data.HRU.cellArea[self.var.land_use_indices_forest]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_forest])
        self.ETref_agriculture[:] = sum(self.var.ETRef[self.var.land_use_indices_agriculture] * self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture])
        self.ETref_grassland[:] = sum(self.var.ETRef[self.var.land_use_indices_grassland] * self.model.data.HRU.cellArea[self.var.land_use_indices_grassland]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_grassland])
        self.averagetemp_forest[:] = sum(tas_C[self.var.land_use_indices_forest] * self.model.data.HRU.cellArea[self.var.land_use_indices_forest]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_forest])
        self.averagetemp_agriculture[:] = sum(tas_C[self.var.land_use_indices_agriculture] * self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture])
        self.averagetemp_grassland[:] = sum(tas_C[self.var.land_use_indices_grassland] * self.model.data.HRU.cellArea[self.var.land_use_indices_grassland]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_grassland])
        self.humidity_forest[:] = sum(hurs[self.var.land_use_indices_forest] * self.model.data.HRU.cellArea[self.var.land_use_indices_forest]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_forest])
        self.humidity_agriculture[:] = sum(hurs[self.var.land_use_indices_agriculture] * self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture])
        self.humidity_grassland[:] = sum(hurs[self.var.land_use_indices_grassland] * self.model.data.HRU.cellArea[self.var.land_use_indices_grassland]) / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_grassland])
        

        self.var.area_forest_ref = self.model.data.HRU.cellArea[self.var.land_use_indices_forest] / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_forest])
        self.var.area_agriculture_ref = self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture] / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture])
        self.var.area_grassland_ref = self.model.data.HRU.cellArea[self.var.land_use_indices_grassland] / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_grassland])
        self.var.area_bioarea_ref = self.model.data.HRU.cellArea[self.var.bioarea_ref] / sum(self.model.data.HRU.cellArea[self.var.bioarea_ref])


        return self.ETref_forest, self.ETref_agriculture, self.ETref_grassland, self.averagetemp_forest, self.averagetemp_agriculture, self.averagetemp_grassland,self.humidity_forest,self.humidity_agriculture,self.humidity_grassland
