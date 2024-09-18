# -------------------------------------------------------------------------
# Name:        Land Cover Type module
# Purpose:
#
# Author:      PB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------
import numpy as np
import xarray as xr

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass
from numba import njit
from geb.workflows import TimingModule


@njit(cache=True)
def get_crop_kc_and_root_depths(
    crop_map,
    crop_age_days_map,
    crop_harvest_age_days,
    irrigated_fields,
    crop_stage_data,
    kc_crop_stage,
    rooth_depths,
    init_root_depth=0.01,
):

    kc = np.full_like(crop_map, np.nan, dtype=np.float32)
    root_depth = np.full_like(crop_map, np.nan, dtype=np.float32)
    irrigated_fields = irrigated_fields.astype(
        np.int8
    )  # change dtype to int, so that we can use the boolean array as index

    for i in range(crop_map.size):
        crop = crop_map[i]
        if crop != -1:
            age_days = crop_age_days_map[i]
            harvest_day = crop_harvest_age_days[i]
            assert harvest_day > 0
            crop_progress = age_days * 100 // harvest_day  # for to be integer
            assert crop_progress <= 100
            d1, d2, d3, d4 = crop_stage_data[crop]
            kc1, kc2, kc3 = kc_crop_stage[crop]
            assert d1 + d2 + d3 + d4 == 100
            if crop_progress <= d1:
                field_kc = kc1
            elif crop_progress <= d1 + d2:
                field_kc = kc1 + (crop_progress - d1) * (kc2 - kc1) / d2
            elif crop_progress <= d1 + d2 + d3:
                field_kc = kc2
            else:
                assert crop_progress <= d1 + d2 + d3 + d4
                field_kc = kc2 + (crop_progress - (d1 + d2 + d3)) * (kc3 - kc2) / d4
            assert not np.isnan(field_kc)
            kc[i] = field_kc

            root_depth[i] = (
                init_root_depth
                + age_days * rooth_depths[crop, irrigated_fields[i]] / harvest_day
            )

    return kc, root_depth


class landcoverType(object):
    """
    LAND COVER TYPE

    runs the 6 land cover types through soil procedures

    This routine calls the soil routine for each land cover type


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    maxGWCapRise          influence of capillary rise above groundwater level                               m
    load_initial
    baseflow              simulated baseflow (= groundwater discharge to river)                             m
    waterbalance_module
    coverTypes            land cover types - forest - grassland - irrPaddy - irrNonPaddy - water - sealed   --
    minInterceptCap       Maximum interception read from file for forest and grassland land cover           m
    interceptStor         simulated vegetation interception storage                                         m
    Rain                  Precipitation less snow                                                           m
    SnowMelt              total snow melt from all layers                                                   m
    snowEvap              total evaporation from snow for a snow layers                                     m
    cellArea              Cell area [m²] of each simulated mesh
    soilLayers            Number of soil layers                                                             --
    landcoverSum
    totalET               Total evapotranspiration for each cell including all landcover types              m
    act_SurfaceWaterAbst
    sum_interceptStor     Total of simulated vegetation interception storage including all landcover types  m
    fracVegCover          Fraction of area covered by the corresponding landcover type
    rootFraction1
    soildepth             Thickness of the first soil layer                                                 m
    soildepth12           Total thickness of layer 2 and 3                                                  m
    KSat1
    KSat2
    KSat3
    alpha1
    alpha2
    alpha3
    lambda1
    lambda2
    lambda3
    thetas1
    thetas2
    thetas3
    thetar1
    thetar2
    thetar3
    genuM1
    genuM2
    genuM3
    genuInvM1
    genuInvM2
    genuInvM3
    genuInvN1
    genuInvN2
    genuInvN3
    invAlpha1
    invAlpha2
    invAlpha3
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
    kUnSat3FC
    kunSatFC12
    kunSatFC23
    cropCoefficientNC_fi
    interceptCapNC_filen
    coverFractionNC_file
    w1                    Simulated water storage in the layer 1                                            m
    w2                    Simulated water storage in the layer 2                                            m
    w3                    Simulated water storage in the layer 3                                            m
    topwater              quantity of water above the soil (flooding)                                       m
    sum_topwater          quantity of water on the soil (flooding) (weighted sum for all landcover types)   m
    totalSto              Total soil,snow and vegetation storage for each cell including all landcover typ  m
    SnowCover             snow cover (sum over all layers)                                                  m
    sum_w1
    sum_w2
    sum_w3
    arnoBetaOro
    ElevationStD
    arnoBeta
    adjRoot
    landcoverSumSum
    totAvlWater
    modflow_timestep      Chosen ModFlow model timestep (1day, 7days, 30days…)
    pretotalSto           Previous totalSto                                                                 m
    sum_actTransTotal
    sum_actBareSoilEvap
    sum_interceptEvap
    addtoevapotrans
    sum_runoff            Runoff above the soil, more interflow, including all landcover types              m
    sum_directRunoff
    sum_interflow
    Precipitation         Precipitation (input for the model)                                               m
    GWVolumeVariation
    sum_availWaterInfilt
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model, ElevationStD):
        """
        Initial part of the land cover type module
        Initialise the six land cover types

        * Forest No.0
        * Grasland/non irrigated land No.1
        * Paddy irrigation No.2
        * non-Paddy irrigation No.3
        * Sealed area No.4
        * Water covered area No.5

        And initialize the soil variables
        """
        self.var = model.data.HRU
        self.model = model
        self.crop_farmers = model.agents.crop_farmers

        self.model.coverTypes = [
            "forest",
            "grassland",
            "irrPaddy",
            "irrNonPaddy",
            "sealed",
            "water",
        ]

        self.var.capriseindex = self.var.full_compressed(0, dtype=np.float32)

        self.var.actBareSoilEvap = self.var.full_compressed(0, dtype=np.float32)
        self.var.actTransTotal = self.var.full_compressed(0, dtype=np.float32)

        self.var.soil_layer_height = np.tile(
            self.var.full_compressed(np.nan, dtype=np.float32),
            (self.model.soilLayers, 1),
        )
        self.var.soil_layer_height[0] = 0.05  # the top soil layer always is 5 cm.
        self.var.soil_layer_height[1] = 0.95  # middle layer extends to 100 cm
        self.var.soil_layer_height[2] = 2.00  # bottom layer extends to 300 cm

        self.var.KSat1 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.KSat2 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.KSat3 = self.var.full_compressed(np.nan, dtype=np.float32)
        alpha1 = self.var.full_compressed(np.nan, dtype=np.float32)
        alpha2 = self.var.full_compressed(np.nan, dtype=np.float32)
        alpha3 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.lambda1 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.lambda2 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.lambda3 = self.var.full_compressed(np.nan, dtype=np.float32)
        thetas1 = self.var.full_compressed(np.nan, dtype=np.float32)
        thetas2 = self.var.full_compressed(np.nan, dtype=np.float32)
        thetas3 = self.var.full_compressed(np.nan, dtype=np.float32)
        thetar1 = self.var.full_compressed(np.nan, dtype=np.float32)
        thetar2 = self.var.full_compressed(np.nan, dtype=np.float32)
        thetar3 = self.var.full_compressed(np.nan, dtype=np.float32)

        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            # TODO: Change soil parameters in forests and grasslands
            land_use_indices = np.where(self.var.land_use_type == coverNum)
            # ksat in cm/d-1 -> m/dm
            self.var.KSat1[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/ksat1"]
                )
                / 100,
                fn=None,
            )[land_use_indices]
            self.var.KSat2[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/ksat2"]
                )
                / 100,
                fn=None,
            )[land_use_indices]
            self.var.KSat3[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/ksat3"]
                )
                / 100,
                fn=None,
            )[land_use_indices]
            alpha1[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/alpha1"]
                ),
                fn=None,
            )[land_use_indices]
            alpha2[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/alpha2"]
                ),
                fn=None,
            )[land_use_indices]
            alpha3[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/alpha3"]
                ),
                fn=None,
            )[land_use_indices]
            self.var.lambda1[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/lambda1"]
                ),
                fn=None,
            )[land_use_indices]
            self.var.lambda2[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/lambda2"]
                ),
                fn=None,
            )[land_use_indices]
            self.var.lambda3[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/lambda3"]
                ),
                fn=None,
            )[land_use_indices]
            thetas1[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/thetas1"]
                ),
                fn=None,
            )[land_use_indices]
            thetas2[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/thetas2"]
                ),
                fn=None,
            )[land_use_indices]
            thetas3[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/thetas3"]
                ),
                fn=None,
            )[land_use_indices]
            thetar1[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/thetar1"]
                ),
                fn=None,
            )[land_use_indices]
            thetar2[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/thetar2"]
                ),
                fn=None,
            )[land_use_indices]
            thetar3[land_use_indices] = self.model.data.to_HRU(
                data=self.model.data.grid.load(
                    self.model.model_structure["grid"]["soil/thetar3"]
                ),
                fn=None,
            )[land_use_indices]

        self.var.wwp1 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wwp2 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wwp3 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.ws1 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.ws2 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.ws3 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wres1 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wres2 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wres3 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wfc1 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wfc2 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wfc3 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wwp1 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wwp2 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.wwp3 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.kunSatFC12 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.kunSatFC23 = self.var.full_compressed(np.nan, dtype=np.float32)

        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            land_use_indices = np.where(self.var.land_use_type == coverNum)
            self.var.ws1[land_use_indices] = (
                thetas1[land_use_indices]
                * self.var.soil_layer_height[0, land_use_indices]
            )
            self.var.ws2[land_use_indices] = (
                thetas2[land_use_indices]
                * self.var.soil_layer_height[1, land_use_indices]
            )
            self.var.ws3[land_use_indices] = (
                thetas3[land_use_indices]
                * self.var.soil_layer_height[2, land_use_indices]
            )

            self.var.wres1[land_use_indices] = (
                thetar1[land_use_indices]
                * self.var.soil_layer_height[0, land_use_indices]
            )
            self.var.wres2[land_use_indices] = (
                thetar2[land_use_indices]
                * self.var.soil_layer_height[1, land_use_indices]
            )
            self.var.wres3[land_use_indices] = (
                thetar3[land_use_indices]
                * self.var.soil_layer_height[2, land_use_indices]
            )

            # Soil moisture at field capacity (pF2, 100 cm) [mm water slice]    # Mualem equation (van Genuchten, 1980)
            self.var.wfc1[land_use_indices] = self.var.wres1[land_use_indices] + (
                self.var.ws1[land_use_indices] - self.var.wres1[land_use_indices]
            ) / (
                (
                    1
                    + (alpha1[land_use_indices] * 100)
                    ** (self.var.lambda1[land_use_indices] + 1)
                )
                ** (
                    self.var.lambda1[land_use_indices]
                    / (self.var.lambda1[land_use_indices] + 1)
                )
            )
            self.var.wfc2[land_use_indices] = self.var.wres2[land_use_indices] + (
                self.var.ws2[land_use_indices] - self.var.wres2[land_use_indices]
            ) / (
                (
                    1
                    + (alpha2[land_use_indices] * 100)
                    ** (self.var.lambda2[land_use_indices] + 1)
                )
                ** (
                    self.var.lambda2[land_use_indices]
                    / (self.var.lambda2[land_use_indices] + 1)
                )
            )
            self.var.wfc3[land_use_indices] = self.var.wres3[land_use_indices] + (
                self.var.ws3[land_use_indices] - self.var.wres3[land_use_indices]
            ) / (
                (
                    1
                    + (alpha3[land_use_indices] * 100)
                    ** (self.var.lambda3[land_use_indices] + 1)
                )
                ** (
                    self.var.lambda3[land_use_indices]
                    / (self.var.lambda3[land_use_indices] + 1)
                )
            )

            # Soil moisture at wilting point (pF4.2, 10**4.2 cm) [mm water slice]    # Mualem equation (van Genuchten, 1980)
            self.var.wwp1[land_use_indices] = self.var.wres1[land_use_indices] + (
                self.var.ws1[land_use_indices] - self.var.wres1[land_use_indices]
            ) / (
                (
                    1
                    + (alpha1[land_use_indices] * (10**4.2))
                    ** (self.var.lambda1[land_use_indices] + 1)
                )
                ** (
                    self.var.lambda1[land_use_indices]
                    / (self.var.lambda1[land_use_indices] + 1)
                )
            )
            self.var.wwp2[land_use_indices] = self.var.wres2[land_use_indices] + (
                self.var.ws2[land_use_indices] - self.var.wres2[land_use_indices]
            ) / (
                (
                    1
                    + (alpha2[land_use_indices] * (10**4.2))
                    ** (self.var.lambda2[land_use_indices] + 1)
                )
                ** (
                    self.var.lambda2[land_use_indices]
                    / (self.var.lambda2[land_use_indices] + 1)
                )
            )
            self.var.wwp3[land_use_indices] = self.var.wres3[land_use_indices] + (
                self.var.ws3[land_use_indices] - self.var.wres3[land_use_indices]
            ) / (
                (
                    1
                    + (alpha3[land_use_indices] * (10**4.2))
                    ** (self.var.lambda3[land_use_indices] + 1)
                )
                ** (
                    self.var.lambda3[land_use_indices]
                    / (self.var.lambda3[land_use_indices] + 1)
                )
            )

            satTerm1FC = np.maximum(
                0.0, self.var.wfc1[land_use_indices] - self.var.wres1[land_use_indices]
            ) / (self.var.ws1[land_use_indices] - self.var.wres1[land_use_indices])
            satTerm2FC = np.maximum(
                0.0, self.var.wfc2[land_use_indices] - self.var.wres2[land_use_indices]
            ) / (self.var.ws2[land_use_indices] - self.var.wres2[land_use_indices])
            satTerm3FC = np.maximum(
                0.0, self.var.wfc3[land_use_indices] - self.var.wres3[land_use_indices]
            ) / (self.var.ws3[land_use_indices] - self.var.wres3[land_use_indices])
            kUnSat1FC = (
                self.var.KSat1[land_use_indices]
                * np.sqrt(satTerm1FC)
                * np.square(
                    1
                    - (
                        1
                        - satTerm1FC
                        ** (
                            1
                            / (
                                self.var.lambda1[land_use_indices]
                                / (self.var.lambda1[land_use_indices] + 1)
                            )
                        )
                    )
                    ** (
                        self.var.lambda1[land_use_indices]
                        / (self.var.lambda1[land_use_indices] + 1)
                    )
                )
            )
            kUnSat2FC = (
                self.var.KSat2[land_use_indices]
                * np.sqrt(satTerm2FC)
                * np.square(
                    1
                    - (
                        1
                        - satTerm2FC
                        ** (
                            1
                            / (
                                self.var.lambda2[land_use_indices]
                                / (self.var.lambda2[land_use_indices] + 1)
                            )
                        )
                    )
                    ** (
                        self.var.lambda2[land_use_indices]
                        / (self.var.lambda2[land_use_indices] + 1)
                    )
                )
            )
            kUnSat3FC = (
                self.var.KSat3[land_use_indices]
                * np.sqrt(satTerm3FC)
                * np.square(
                    1
                    - (
                        1
                        - satTerm3FC
                        ** (
                            1
                            / (
                                self.var.lambda3[land_use_indices]
                                / (self.var.lambda3[land_use_indices] + 1)
                            )
                        )
                    )
                    ** (
                        self.var.lambda3[land_use_indices]
                        / (self.var.lambda3[land_use_indices] + 1)
                    )
                )
            )
            self.var.kunSatFC12[land_use_indices] = np.sqrt(kUnSat1FC * kUnSat2FC)
            self.var.kunSatFC23[land_use_indices] = np.sqrt(kUnSat2FC * kUnSat3FC)

        # for paddy irrigation flooded paddy fields
        self.var.topwater = self.model.data.HRU.load_initial(
            "topwater", default=self.var.full_compressed(0, dtype=np.float32)
        )

        self.var.arnoBeta = self.var.full_compressed(np.nan, dtype=np.float32)

        # Improved Arno's scheme parameters: Hageman and Gates 2003
        # arnoBeta defines the shape of soil water capacity distribution curve as a function of  topographic variability
        # b = max( (oh - o0)/(oh + omax), 0.01)
        # oh: the standard deviation of orography, o0: minimum std dev, omax: max std dev
        arnoBetaOro = (ElevationStD - 10.0) / (ElevationStD + 1500.0)

        # for CALIBRATION
        arnoBetaOro = arnoBetaOro + self.model.config["parameters"]["arnoBeta_add"]
        arnoBetaOro = np.minimum(1.2, np.maximum(0.01, arnoBetaOro))

        initial_humidy = 0.5
        self.var.w1 = self.model.data.HRU.load_initial(
            "w1",
            default=np.nan_to_num(
                self.var.wwp1 + initial_humidy * (self.var.wfc1 - self.var.wwp1)
            ),
        )
        self.var.w2 = self.model.data.HRU.load_initial(
            "w2",
            default=np.nan_to_num(
                self.var.wwp2 + initial_humidy * (self.var.wfc2 - self.var.wwp2)
            ),
        )
        self.var.w3 = self.model.data.HRU.load_initial(
            "w3",
            default=np.nan_to_num(
                self.var.wwp3 + initial_humidy * (self.var.wfc3 - self.var.wwp3)
            ),
        )

        arnobeta_cover_types = {
            "forest": 0.2,
            "grassland": 0.0,
            "irrPaddy": 0.2,
            "irrNonPaddy": 0.2,
        }

        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            land_use_indices = np.where(self.var.land_use_type == coverNum)[0]

            arnoBeta = arnobeta_cover_types[coverType]
            if not isinstance(arnoBeta, float):
                arnoBeta = arnoBeta[land_use_indices]
            self.var.arnoBeta[land_use_indices] = (arnoBetaOro + arnoBeta)[
                land_use_indices
            ]
            self.var.arnoBeta[land_use_indices] = np.minimum(
                1.2, np.maximum(0.01, self.var.arnoBeta[land_use_indices])
            )

        self.forest_kc_per_10_days = xr.open_dataset(
            self.model.model_structure["forcing"][
                "landcover/forest/cropCoefficientForest_10days"
            ]
        )["cropCoefficientForest_10days"].values

    def water_body_exchange(self, groundwater_recharge):
        """computing leakage from rivers"""
        riverbedExchangeM3 = (
            self.model.data.grid.leakageriver_factor
            * self.var.cellArea
            * ((1 - self.var.capriseindex + 0.25) // 1)
        )
        riverbedExchangeM3[self.var.land_use_type != 5] = 0
        riverbedExchangeM3 = self.model.data.to_grid(
            HRU_data=riverbedExchangeM3, fn="sum"
        )
        riverbedExchangeM3 = np.minimum(
            riverbedExchangeM3, 0.80 * self.model.data.grid.channelStorageM3
        )
        # if there is a lake in this cell, there is no leakage
        riverbedExchangeM3[self.model.data.grid.waterBodyID > 0] = 0

        # adding leakage from river to the groundwater recharge
        waterbed_recharge = self.model.data.grid.M3toM(riverbedExchangeM3)

        # riverbed exchange means water is being removed from the river to recharge
        self.model.data.grid.riverbedExchangeM3 = (
            riverbedExchangeM3  # to be used in routing_kinematic
        )

        # first, lakes variable need to be extended to their area and not only to the discharge point
        lakeIDbyID = np.unique(self.model.data.grid.waterBodyID)

        lakestor_id = np.copy(self.model.data.grid.lakeStorage)
        resstor_id = np.copy(self.model.data.grid.resStorage)
        for id in range(len(lakeIDbyID)):  # for each lake or reservoir
            if lakeIDbyID[id] != 0:
                temp_map = np.where(
                    self.model.data.grid.waterBodyID == lakeIDbyID[id],
                    np.where(self.model.data.grid.lakeStorage > 0, 1, 0),
                    0,
                )  # Looking for the discharge point of the lake
                if np.sum(temp_map) == 0:  # try reservoir
                    temp_map = np.where(
                        self.model.data.grid.waterBodyID == lakeIDbyID[id],
                        np.where(self.model.data.grid.resStorage > 0, 1, 0),
                        0,
                    )  # Looking for the discharge point of the reservoir
                discharge_point = np.nanargmax(
                    temp_map
                )  # Index of the cell where the lake outlet is stored
                if self.model.data.grid.waterBodyTypTemp[discharge_point] != 0:

                    if (
                        self.model.data.grid.waterBodyTypTemp[discharge_point] == 1
                    ):  # this is a lake
                        # computing the lake area
                        area_stor = np.sum(
                            np.where(
                                self.model.data.grid.waterBodyID == lakeIDbyID[id],
                                self.model.data.grid.cellArea,
                                0,
                            )
                        )  # required to keep mass balance rigth
                        # computing the lake storage in meter and put this value in each cell including the lake
                        lakestor_id = np.where(
                            self.model.data.grid.waterBodyID == lakeIDbyID[id],
                            self.model.data.grid.lakeStorage[discharge_point]
                            / area_stor,
                            lakestor_id,
                        )  # in meter

                    else:  # this is a reservoir
                        # computing the reservoir area
                        area_stor = np.sum(
                            np.where(
                                self.model.data.grid.waterBodyID == lakeIDbyID[id],
                                self.model.data.grid.cellArea,
                                0,
                            )
                        )  # required to keep mass balance rigth
                        # computing the reservoir storage in meter and put this value in each cell including the reservoir
                        resstor_id = np.where(
                            self.model.data.grid.waterBodyID == lakeIDbyID[id],
                            self.model.data.grid.resStorage[discharge_point]
                            / area_stor,
                            resstor_id,
                        )  # in meter

        # Gathering lakes and reservoirs in the same array
        lakeResStorage = np.where(
            self.model.data.grid.waterBodyTypTemp == 0,
            0.0,
            np.where(
                self.model.data.grid.waterBodyTypTemp == 1, lakestor_id, resstor_id
            ),
        )  # in meter

        minlake = np.maximum(
            0.0, 0.98 * lakeResStorage
        )  # reasonable but arbitrary limit

        # leakage depends on water bodies storage, water bodies fraction and modflow saturated area
        lakebedExchangeM = self.model.data.grid.leakagelake_factor * (
            (1 - self.var.capriseindex + 0.25) // 1
        )
        lakebedExchangeM[self.var.land_use_type != 5] = 0
        lakebedExchangeM = self.model.data.to_grid(HRU_data=lakebedExchangeM, fn="sum")
        lakebedExchangeM = np.minimum(lakebedExchangeM, minlake)

        # Now, leakage is converted again from the lake/reservoir area to discharge point to be removed from the lake/reservoir store
        self.model.data.grid.lakebedExchangeM3 = np.zeros(
            self.model.data.grid.compressed_size, dtype=np.float32
        )
        for id in range(len(lakeIDbyID)):  # for each lake or reservoir
            if lakeIDbyID[id] != 0:
                temp_map = np.where(
                    self.model.data.grid.waterBodyID == lakeIDbyID[id],
                    np.where(self.model.data.grid.lakeStorage > 0, 1, 0),
                    0,
                )  # Looking for the discharge point of the lake
                if np.sum(temp_map) == 0:  # try reservoir
                    temp_map = np.where(
                        self.model.data.grid.waterBodyID == lakeIDbyID[id],
                        np.where(self.model.data.grid.resStorage > 0, 1, 0),
                        0,
                    )  # Looking for the discharge point of the reservoir
                discharge_point = np.nanargmax(
                    temp_map
                )  # Index of the cell where the lake outlet is stored
            # Converting the lake/reservoir leakage from meter to cubic meter and put this value in the cell corresponding to the outlet
            self.model.data.grid.lakebedExchangeM3[discharge_point] = np.sum(
                np.where(
                    self.model.data.grid.waterBodyID == lakeIDbyID[id],
                    lakebedExchangeM * self.model.data.grid.cellArea,
                    0,
                )
            )  # in m3
        self.model.data.grid.lakebedExchangeM = self.model.data.grid.M3toM(
            self.model.data.grid.lakebedExchangeM3
        )

        # compressed version for lakes and reservoirs
        lakeExchangeM3 = (
            np.compress(
                self.model.data.grid.compress_LR, self.model.data.grid.lakebedExchangeM
            )
            * self.model.data.grid.MtoM3C
        )

        # substract from both, because it is sorted by self.var.waterBodyTypCTemp
        self.model.data.grid.lakeStorageC = (
            self.model.data.grid.lakeStorageC - lakeExchangeM3
        )
        # assert (self.model.data.grid.lakeStorageC >= 0).all()
        self.model.data.grid.lakeVolumeM3C = (
            self.model.data.grid.lakeVolumeM3C - lakeExchangeM3
        )
        self.model.data.grid.reservoirStorageM3C = (
            self.model.data.grid.reservoirStorageM3C - lakeExchangeM3
        )

        # and from the combined one for waterbalance issues
        self.model.data.grid.lakeResStorageC = (
            self.model.data.grid.lakeResStorageC - lakeExchangeM3
        )
        # assert (self.model.data.grid.lakeResStorageC >= 0).all()
        self.model.data.grid.lakeResStorage = self.model.data.grid.full_compressed(
            0, dtype=np.float32
        )
        np.put(
            self.model.data.grid.lakeResStorage,
            self.model.data.grid.decompress_LR,
            self.model.data.grid.lakeResStorageC,
        )

        # adding leakage from lakes and reservoirs to the groundwater recharge
        waterbed_recharge += lakebedExchangeM

        groundwater_recharge += waterbed_recharge

    def step(self):
        """
        Dynamic part of the land cover type module

        Calculating soil for each of the 6  land cover class

        * calls evaporation_module.dynamic
        * calls interception_module.dynamic
        * calls soil_module.dynamic
        * calls sealed_water_module.dynamic

        And sums every thing up depending on the land cover type fraction
        """

        timer = TimingModule("Landcover")

        if self.model.CHECK_WATER_BALANCE:
            interceptStor_pre = self.var.interceptStor.copy()
            w1_pre = self.var.w1.copy()
            w2_pre = self.var.w2.copy()
            w3_pre = self.var.w3.copy()
            topwater_pre = self.var.topwater.copy()

        crop_stage_lenghts = np.column_stack(
            [
                self.crop_farmers.crop_data["l_ini"],
                self.crop_farmers.crop_data["l_dev"],
                self.crop_farmers.crop_data["l_mid"],
                self.crop_farmers.crop_data["l_late"],
            ]
        )

        crop_factors = np.column_stack(
            [
                self.crop_farmers.crop_data["kc_initial"],
                self.crop_farmers.crop_data["kc_mid"],
                self.crop_farmers.crop_data["kc_end"],
            ]
        )

        root_depths = np.column_stack(
            [
                self.crop_farmers.crop_data["rd_rain"],
                self.crop_farmers.crop_data["rd_irr"],
            ]
        )

        self.var.cropKC, self.var.root_depth = get_crop_kc_and_root_depths(
            self.var.crop_map,
            self.var.crop_age_days_map,
            self.var.crop_harvest_age_days,
            irrigated_fields=self.model.agents.crop_farmers.irrigated_fields,
            crop_stage_data=crop_stage_lenghts,
            kc_crop_stage=crop_factors,
            rooth_depths=root_depths,
            init_root_depth=0.01,
        )

        self.var.root_depth[self.var.land_use_type == 0] = 2.0  # forest
        self.var.root_depth[
            (self.var.land_use_type == 1) & (self.var.land_owners == -1)
        ] = 0.1  # grassland
        self.var.root_depth[
            (self.var.land_use_type == 1) & (self.var.land_owners != -1)
        ] = 0.05  # fallow land. The rooting depth

        if self.model.use_gpu:
            self.var.cropKC = cp.array(self.var.cropKC)

        forest_cropCoefficientNC = self.model.data.to_HRU(
            data=self.model.data.grid.compress(
                self.forest_kc_per_10_days[(self.model.current_day_of_year - 1) // 10]
            ),
            fn=None,
        )

        self.var.cropKC[self.var.land_use_type == 0] = forest_cropCoefficientNC[
            self.var.land_use_type == 0
        ]  # forest
        assert (self.var.crop_map[self.var.land_use_type == 1] == -1).all()

        self.var.cropKC[self.var.land_use_type == 1] = 0.2

        self.var.potTranspiration, potBareSoilEvap, totalPotET = (
            self.model.evaporation_module.step(self.var.ETRef)
        )

        potTranspiration_minus_interception_evaporation = (
            self.model.interception_module.step(self.var.potTranspiration)
        )  # first thing that evaporates is the intercepted water.
        timer.new_split("Transpiration")

        # *********  WATER Demand   *************************
        groundwater_abstaction, channel_abstraction_m, addtoevapotrans, returnFlow = (
            self.model.waterdemand_module.step(totalPotET)
        )
        timer.new_split("Demand")

        openWaterEvap = self.var.full_compressed(0, dtype=np.float32)
        # Soil for forest, grassland, and irrigated land
        capillar = self.model.data.to_HRU(data=self.model.data.grid.capillar, fn=None)
        del self.model.data.grid.capillar

        (
            interflow,
            directRunoff,
            groundwater_recharge,
            openWaterEvap,
        ) = self.model.soil_module.step(
            capillar,
            openWaterEvap,
            potTranspiration_minus_interception_evaporation,
            potBareSoilEvap,
            totalPotET,
        )
        timer.new_split("Soil")

        directRunoff = self.model.sealed_water_module.step(
            capillar, openWaterEvap, directRunoff
        )
        timer.new_split("Sealed")

        if self.model.use_gpu:
            self.var.actual_transpiration_crop[
                self.var.crop_map != -1
            ] += self.var.actTransTotal.get()[self.var.crop_map != -1]
            self.var.potential_transpiration_crop[
                self.var.crop_map != -1
            ] += self.var.potTranspiration.get()[self.var.crop_map != -1]
        else:
            # print(
            #     "act",
            #     self.var.actTransTotal[self.var.crop_map != -1].mean(),
            #     self.var.actTransTotal[self.var.crop_map != -1].min(),
            #     self.var.actTransTotal[self.var.crop_map != -1].max(),
            # )
            self.var.actual_transpiration_crop[
                self.var.crop_map != -1
            ] += self.var.actTransTotal[self.var.crop_map != -1]
            self.var.potential_transpiration_crop[
                self.var.crop_map != -1
            ] += self.var.potTranspiration[self.var.crop_map != -1]

        assert not np.isnan(interflow).any()
        assert not np.isnan(groundwater_recharge).any()
        assert not np.isnan(groundwater_abstaction).any()
        assert not np.isnan(channel_abstraction_m).any()
        assert not np.isnan(openWaterEvap).any()

        if self.model.CHECK_WATER_BALANCE:
            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[self.var.Rain, self.var.SnowMelt],
                outfluxes=[
                    self.var.natural_available_water_infiltration,
                    self.var.interceptEvap,
                ],
                prestorages=[interceptStor_pre],
                poststorages=[self.var.interceptStor],
                tollerance=1e-6,
            )

            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[
                    self.var.natural_available_water_infiltration,
                    capillar,
                    self.var.actual_irrigation_consumption,
                ],
                outfluxes=[
                    directRunoff,
                    interflow,
                    groundwater_recharge,
                    self.var.actTransTotal,
                    self.var.actBareSoilEvap,
                    openWaterEvap,
                ],
                prestorages=[w1_pre, w2_pre, w3_pre, topwater_pre],
                poststorages=[self.var.w1, self.var.w2, self.var.w3, self.var.topwater],
                tollerance=1e-6,
            )

            totalstorage = (
                np.sum(self.var.SnowCoverS, axis=0)
                / self.model.snowfrost_module.numberSnowLayers
                + self.var.interceptStor
                + self.var.w1
                + self.var.w2
                + self.var.w3
                + self.var.topwater
            )
            totalstorage_pre = (
                self.var.prevSnowCover
                + w1_pre
                + w2_pre
                + w3_pre
                + topwater_pre
                + interceptStor_pre
            )

            self.model.waterbalance_module.waterBalanceCheck(
                how="cellwise",
                influxes=[
                    self.var.precipitation_m_day,
                    self.var.actual_irrigation_consumption,
                    capillar,
                ],
                outfluxes=[
                    directRunoff,
                    interflow,
                    groundwater_recharge,
                    self.var.actTransTotal,
                    self.var.actBareSoilEvap,
                    openWaterEvap,
                    self.var.interceptEvap,
                    self.var.snowEvap,
                ],
                prestorages=[totalstorage_pre],
                poststorages=[totalstorage],
                tollerance=1e-6,
            )

        groundwater_recharge = self.model.data.to_grid(
            HRU_data=groundwater_recharge, fn="weightedmean"
        )
        # self.water_body_exchange(groundwater_recharge)

        timer.new_split("Waterbody exchange")

        if self.model.timing:
            print(timer)

        return (
            self.model.data.to_grid(HRU_data=interflow, fn="weightedmean"),
            self.model.data.to_grid(HRU_data=directRunoff, fn="weightedmean"),
            groundwater_recharge,
            groundwater_abstaction,
            channel_abstraction_m,
            openWaterEvap,
            returnFlow,
        )