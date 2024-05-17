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
import pandas as pd
import xarray as xr
import rioxarray
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    pass
from numba import njit
from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import checkOption, binding, loadmap, cbinding


@njit(cache=True)
def get_crop_kc(crop_map, crop_age_days_map, crop_harvest_age_days, crop_stage_data, kc_crop_stage):
    assert (kc_crop_stage != 0).all()
    assert (crop_stage_data != 0).all()

    shape = crop_map.shape
    crop_map = crop_map.ravel()
    crop_age_days_map = crop_age_days_map.ravel()
    
    kc = np.full(crop_map.size, np.nan, dtype=np.float32)

    for i in range(crop_map.size):
        crop = crop_map[i]
        if crop != -1:
            age_days = crop_age_days_map[i]
            harvest_day = crop_harvest_age_days[i]
            assert harvest_day > 0
            crop_progress = age_days / harvest_day * 100
            d1, d2, d3, d4 = crop_stage_data[crop]
            kc1, kc2, kc3 = kc_crop_stage[crop]
            assert d1 + d2 + d3 + d4 == 100
            if crop_progress < d1:
                field_kc = kc1
            elif crop_progress < d1 + d2:
                field_kc = kc1 + (crop_progress - d1) * (kc2 - kc1) / d2
            elif crop_progress < d1 + d2 + d3:
                field_kc = kc2
            else:
                assert crop_progress <= d1 + d2 + d3 + d4
                field_kc = kc2 + (crop_progress - (d1 + d2 + d3)) * (kc3 - kc2) / d4
            assert not np.isnan(field_kc)
            kc[i] = field_kc
    
    return kc.reshape(shape)


class landcoverType(object):

    """
    LAND COVER TYPE

    runs the 6 land cover types through soil procedures

    This routine calls the soil routine for each land cover type


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    modflow               Flag: True if modflow_coupling = True in settings file                            --       
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
    dynamicLandcover                                                                                                 
    soilLayers            Number of soil layers                                                             --       
    landcoverSum                                                                                                     
    totalET               Total evapotranspiration for each cell including all landcover types              m        
    act_SurfaceWaterAbst                                                                                             
    sum_interceptStor     Total of simulated vegetation interception storage including all landcover types  m        
    fracVegCover          Fraction of area covered by the corresponding landcover type                               
    minCropKC             minimum crop factor (default 0.2)                                                 --       
    rootFraction1                                                                                                    
    maxRootDepth                                                                                                     
    rootDepth                                                                                                        
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
    maxtopwater           maximum heigth of topwater                                                        m        
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

    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model
        self.farmers = model.agents.farmers

    def initial(self, ElevationStD):
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

        self.model.coverTypes = ["forest", "grassland", "irrPaddy", "irrNonPaddy", "sealed", "water"]

        if self.model.config["general"]["name"] == "100 infiltration change" or self.model.config["general"]["name"] == "100 no infiltration change": #self.model.config["general"]["name"] == "spinup"
                # Create a mask for areas with value 1 in the raster, everything else = 0 
                to_forest =  rioxarray.open_rasterio("C:/Users/romij/GEB/GEB_models/meuse/models/meuse/base/input/to_forest/forested_grassland_and_agricultural_land.tif", masked = True)

        elif self.model.config["general"]["name"] == "restoration opportunities":
                to_forest =  rioxarray.open_rasterio("C:/Users/romij/GEB/GEB_models/meuse/models/meuse/base/input/to_forest/reclass_and_reprojected_restoration_opportunities.tif", masked = True)

        if self.model.config["general"]["name"] == "100 infiltration change" or self.model.config["general"]["name"] == "100 no infiltration change" or self.model.config["general"]["name"] == "restoration opportunities" :
                forest_mask_3d = np.where(to_forest.values == 1,1,0)
                forest_mask_3d_boolean = forest_mask_3d == 1
                forest_mask = forest_mask_3d_boolean[0, :, :]

                HRU_indices = self.var.decompress(
                    np.arange(self.model.data.HRU.land_use_type.size)
                )
                HRUs_to_forest = np.unique(HRU_indices[forest_mask])
                HRUs_to_forest = HRUs_to_forest[HRUs_to_forest != -1]
                HRUs_to_forest = HRUs_to_forest[
                    (self.var.land_use_type[HRUs_to_forest] >= 1) & 
                    (self.var.land_use_type[HRUs_to_forest] <= 3)
                ]  # select HRUs which are grassland or agricultural land 
                
                self.var.land_use_type[HRUs_to_forest] = 0  # 0 is forest
                
                # Define the range of values
                range_values = range(6)  # Values from 0 to 5

                # Initialize a dictionary to store counts for each value
                counts = {}

                # Count occurrences of each value
                for value in range_values:
                    counts[value] = np.count_nonzero(self.var.land_use_type == value)

                # Calculate percentage for each value
                total_elements = self.var.land_use_type.size
                percentages = {value: (count / total_elements) * 100 for value, count in counts.items()}

                # Print the results
                for value, percentage in percentages.items():
                    print(f"The percentage of {value} occurring in the array is: {percentage:.2f}%")

                # Change values of 2 and 3 to 0
                #self.var.land_use_type[self.var.land_use_type == 2] = 0
                #self.var.land_use_type[self.var.land_use_type == 3] = 0

        #extract land use types for land use map

        #self.var.decompress(
                   # np.arange(self.model.data.HRU.land_use_type.size)
                #)
        # transform = Affine.from_gdal(*self.model.data.HRU.gt)
        # self.model.industry_water_consumption_ds.rio.transform().to_gdal()
 
                # Create a new mask that includes the areas to be converted to forest
                #forest_mask = geometry_mask(
                 #   [geom for geom in to_forest.geometry],
                  #  transform=transform,
                   # out_shape=self.model.data.HRU.mask.shape,
                    #invert=True,
                    #all_touched=True,
                #)

   
        mask = ((self.var.land_use_type == 1) & (self.var.land_owners != -1)) #change land use type grassland to agriculture where land owners are
        self.var.land_use_type[mask] = 3
        self.var.land_use_indices_forest = np.where(self.var.land_use_type == 0) 
        self.var.land_use_indices_grassland = np.where(self.var.land_use_type == 1) 
        self.var.land_use_indices_agriculture = np.where((self.var.land_use_type == 2) | (self.var.land_use_type == 3))
        self.var.bioarea_ref = np.where(self.var.land_use_type < 4)[0].astype(np.int32)

        self.var.capriseindex = self.var.full_compressed(0, dtype=np.float32)

        self.var.actBareSoilEvap = self.var.full_compressed(0, dtype=np.float32)
        self.var.actTransTotal = self.var.full_compressed(0, dtype=np.float32)

        self.var.minCropKC = loadmap('minCropKC')

        rootFraction1 = self.var.full_compressed(np.nan, dtype=np.float32)
        maxRootDepth = self.var.full_compressed(np.nan, dtype=np.float32)
        soildepth_factor = loadmap('soildepth_factor')
        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            land_use_indices = np.where(self.var.land_use_type == coverNum)
            rootFraction1[land_use_indices] = self.model.data.to_HRU(data=loadmap(coverType + "_rootFraction1"), fn=None)[land_use_indices]
            maxRootDepth[land_use_indices] = self.model.data.to_HRU(data=loadmap(coverType + "_maxRootDepth") * soildepth_factor, fn=None)[land_use_indices]

        self.var.rootDepth1 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.rootDepth2 = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.rootDepth3 = self.var.full_compressed(np.nan, dtype=np.float32)
        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            land_use_indices = np.where(self.var.land_use_type == coverNum)
            # calculate rootdepth for each soillayer and each land cover class
            self.var.rootDepth1[land_use_indices] = self.var.soildepth[0][land_use_indices]  # 0.05 m
            h1 = np.maximum(self.var.soildepth[1][land_use_indices], maxRootDepth[land_use_indices] - self.var.soildepth[0][land_use_indices])
            #
            self.var.rootDepth2[land_use_indices] = np.minimum(self.var.soildepth[1][land_use_indices] + self.var.soildepth[2][land_use_indices] - 0.05, h1)
            # soil layer is minimum 0.05 m
            self.var.rootDepth3[land_use_indices] = np.maximum(0.05, self.var.soildepth[1][land_use_indices] + self.var.soildepth[2][land_use_indices] - self.var.rootDepth2[land_use_indices]) # What is the motivation of this pushing the roodDepth[1] to the maxRotDepth

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


        self.var.area_forest_ref = self.model.data.HRU.cellArea[self.var.land_use_indices_forest] / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_forest])
        self.var.area_agriculture_ref = self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture] / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_agriculture])
        self.var.area_grassland_ref = self.model.data.HRU.cellArea[self.var.land_use_indices_grassland] / sum(self.model.data.HRU.cellArea[self.var.land_use_indices_grassland])
        self.var.area_bioarea_ref = self.model.data.HRU.cellArea[self.var.bioarea_ref] / sum(self.model.data.HRU.cellArea[self.var.bioarea_ref])

        if (self.model.config["general"]["name"] == "100 infiltration change" or \
        self.model.config["general"]["name"] == "infiltration change no lulc" or \
        self.model.config["general"]["name"] == "spinup" or \
        self.model.config["general"]["name"] == "restoration opportunities"):

            for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
                land_use_indices = np.where(self.var.land_use_type == coverNum)
                if coverType == 'forest':
                    # for forest there is a special map, for the other land use types the same map is used
                        # ksat in cm/d-1 -> m/dm
                    self.var.KSat1[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_ksat1")/100, fn=None)[land_use_indices]  # checked
                    self.var.KSat2[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_ksat2")/100, fn=None)[land_use_indices]  # checked
                    self.var.KSat3[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_ksat3")/100, fn=None)[land_use_indices]  # checked
                    alpha1[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_alpha1"), fn=None)[land_use_indices]  # checked
                    alpha2[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_alpha2"), fn=None)[land_use_indices]  # checked
                    alpha3[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_alpha3"), fn=None)[land_use_indices]  # checked
                    self.var.lambda1[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_lambda1"), fn=None)[land_use_indices]  # checked
                    self.var.lambda2[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_lambda2"), fn=None)[land_use_indices]  # checked
                    self.var.lambda3[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_lambda3"), fn=None)[land_use_indices]  # checked
                    thetas1[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_thetas1"), fn=None)[land_use_indices]  # checked
                    thetas2[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_thetas2"), fn=None)[land_use_indices]  # checked
                    thetas3[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_thetas3"), fn=None)[land_use_indices]  # checked
                    thetar1[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_thetar1"), fn=None)[land_use_indices]  # checked
                    thetar2[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_thetar2"), fn=None)[land_use_indices]  # checked
                    thetar3[land_use_indices] = self.model.data.to_HRU(data=loadmap("forest_fao_thetar3"), fn=None)[land_use_indices]  # checked

                else:
                    pre = ""                  
                    self.var.KSat1[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "KSat1")/100, fn=None)[land_use_indices]  # checked
                    self.var.KSat2[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "KSat2")/100, fn=None)[land_use_indices]  # checked
                    self.var.KSat3[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "KSat3")/100, fn=None)[land_use_indices]  # checked
                    alpha1[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "alpha1"), fn=None)[land_use_indices]  # checked
                    alpha2[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "alpha2"), fn=None)[land_use_indices]  # checked
                    alpha3[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "alpha3"), fn=None)[land_use_indices]  # checked
                    self.var.lambda1[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "lambda1"), fn=None)[land_use_indices]  # checked
                    self.var.lambda2[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "lambda2"), fn=None)[land_use_indices]  # checked
                    self.var.lambda3[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "lambda3"), fn=None)[land_use_indices]  # checked
                    thetas1[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "thetas1"), fn=None)[land_use_indices]  # checked
                    thetas2[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "thetas2"), fn=None)[land_use_indices]  # checked
                    thetas3[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "thetas3"), fn=None)[land_use_indices]  # checked
                    thetar1[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "thetar1"), fn=None)[land_use_indices]  # checked
                    thetar2[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "thetar2"), fn=None)[land_use_indices]  # checked
                    thetar3[land_use_indices] = self.model.data.to_HRU(data=loadmap(pre + "thetar3"), fn=None)[land_use_indices]  # checked

                if coverType == 'grassland':
                    data_ksat2 = self.model.data.to_HRU(data=loadmap("grs_fao_ksat2")/100, fn=None)
                    nonzero_mask = data_ksat2 != 0
                    true_locations = np.where(nonzero_mask[land_use_indices])[0]
                    data = self.model.data.to_HRU(data=loadmap("grs_fao_ksat2")/100, fn=None)[land_use_indices]
                    grs_ksat = self.var.KSat2[land_use_indices]
                    grs_ksat[true_locations] = data[true_locations]
                    self.var.KSat2[land_use_indices] = grs_ksat

                if coverType == 'irrNonPaddy':
                    self.var.KSat1[land_use_indices] = self.model.data.to_HRU(data=loadmap("KSat1")/100, fn=None)[land_use_indices] 
                    #values filled in fao polygons are filled for every parameter, without transferring polygons which are 0 in fao map to the rest of the values; essentially not overwriting all values already filled in by the original data above
                    parameters = ["KSat1", "KSat2", "KSat3", "alpha1", "alpha2", "alpha3", "lambda1", "lambda2", "lambda3", "thetas1", "thetas2", "thetas3", "thetar1", "thetar2", "thetar3"]
                    for param in parameters:
                        # Load data and create a boolean mask for non-zero values
                        data_param = self.model.data.to_HRU(data=loadmap("agr_fao_" + param), fn=None)
                        nonzero_mask = data_param != 0

                        # Find indices where both land_use_indices and nonzero_mask are True
                        true_locations = np.where(nonzero_mask[land_use_indices])[0]

                        # Get the data and current parameter values corresponding to land_use_indices
                        data = self.model.data.to_HRU(data=loadmap("agr_fao_" + param), fn=None)[land_use_indices]
                        param_values = getattr(self.var, param)[land_use_indices] if hasattr(self.var, param) else locals()[param][land_use_indices]

                        # Check if the parameter is KSat1, KSat2, or KSat3 and divide the data by 100 if true
                        if param.startswith("KSat"):
                            data /= 100

                        # Update parameter values at true_locations
                        param_values[true_locations] = data[true_locations]

                        # Assign updated parameter values back to the class attribute or local variable
                        if hasattr(self.var, param):
                            getattr(self.var, param)[land_use_indices] = param_values
                        else:
                            locals()[param][land_use_indices] = param_values

                            
        else:
                for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
                    land_use_indices = np.where(self.var.land_use_type == coverNum)          
                    self.var.KSat1[land_use_indices] = self.model.data.to_HRU(data=loadmap("KSat1")/100, fn=None)[land_use_indices]  # checked
                    self.var.KSat2[land_use_indices] = self.model.data.to_HRU(data=loadmap("KSat2")/100, fn=None)[land_use_indices]  # checked
                    self.var.KSat3[land_use_indices] = self.model.data.to_HRU(data=loadmap("KSat3")/100, fn=None)[land_use_indices]  # checked
                    alpha1[land_use_indices] = self.model.data.to_HRU(data=loadmap("alpha1"), fn=None)[land_use_indices]  # checked
                    alpha2[land_use_indices] = self.model.data.to_HRU(data=loadmap("alpha2"), fn=None)[land_use_indices]  # checked
                    alpha3[land_use_indices] = self.model.data.to_HRU(data=loadmap("alpha3"), fn=None)[land_use_indices]  # checked
                    self.var.lambda1[land_use_indices] = self.model.data.to_HRU(data=loadmap("lambda1"), fn=None)[land_use_indices]  # checked
                    self.var.lambda2[land_use_indices] = self.model.data.to_HRU(data=loadmap("lambda2"), fn=None)[land_use_indices]  # checked
                    self.var.lambda3[land_use_indices] = self.model.data.to_HRU(data=loadmap("lambda3"), fn=None)[land_use_indices]  # checked
                    thetas1[land_use_indices] = self.model.data.to_HRU(data=loadmap("thetas1"), fn=None)[land_use_indices]  # checked
                    thetas2[land_use_indices] = self.model.data.to_HRU(data=loadmap("thetas2"), fn=None)[land_use_indices]  # checked
                    thetas3[land_use_indices] = self.model.data.to_HRU(data=loadmap("thetas3"), fn=None)[land_use_indices]  # checked
                    thetar1[land_use_indices] = self.model.data.to_HRU(data=loadmap("thetar1"), fn=None)[land_use_indices]  # checked
                    thetar2[land_use_indices] = self.model.data.to_HRU(data=loadmap("thetar2"), fn=None)[land_use_indices]  # checked
                    thetar3[land_use_indices] = self.model.data.to_HRU(data=loadmap("thetar3"), fn=None)[land_use_indices]  # checked



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
            self.var.ws1[land_use_indices] = thetas1[land_use_indices] * self.var.rootDepth1[land_use_indices]
            self.var.ws2[land_use_indices] = thetas2[land_use_indices] * self.var.rootDepth2[land_use_indices]
            self.var.ws3[land_use_indices] = thetas3[land_use_indices] * self.var.rootDepth3[land_use_indices]

            self.var.wres1[land_use_indices] = thetar1[land_use_indices] * self.var.rootDepth1[land_use_indices]
            self.var.wres2[land_use_indices] = thetar2[land_use_indices] * self.var.rootDepth2[land_use_indices]
            self.var.wres3[land_use_indices] = thetar3[land_use_indices] * self.var.rootDepth3[land_use_indices]

            # Soil moisture at field capacity (pF2, 100 cm) [mm water slice]    # Mualem equation (van Genuchten, 1980)
            self.var.wfc1[land_use_indices] = self.var.wres1[land_use_indices] + (self.var.ws1[land_use_indices] - self.var.wres1[land_use_indices]) / ((1 + (alpha1[land_use_indices] * 100) ** (self.var.lambda1[land_use_indices] + 1)) ** (self.var.lambda1[land_use_indices] / (self.var.lambda1[land_use_indices] + 1)))
            self.var.wfc2[land_use_indices] = self.var.wres2[land_use_indices] + (self.var.ws2[land_use_indices] - self.var.wres2[land_use_indices]) / ((1 + (alpha2[land_use_indices] * 100) ** (self.var.lambda2[land_use_indices] + 1)) ** (self.var.lambda2[land_use_indices] / (self.var.lambda2[land_use_indices] + 1)))
            self.var.wfc3[land_use_indices] = self.var.wres3[land_use_indices] + (self.var.ws3[land_use_indices] - self.var.wres3[land_use_indices]) / ((1 + (alpha3[land_use_indices] * 100) ** (self.var.lambda3[land_use_indices] + 1)) ** (self.var.lambda3[land_use_indices] / (self.var.lambda3[land_use_indices] + 1)))

            # Soil moisture at wilting point (pF4.2, 10**4.2 cm) [mm water slice]    # Mualem equation (van Genuchten, 1980)
            self.var.wwp1[land_use_indices] = self.var.wres1[land_use_indices] + (self.var.ws1[land_use_indices] - self.var.wres1[land_use_indices]) / ((1 + (alpha1[land_use_indices] * (10**4.2)) ** (self.var.lambda1[land_use_indices] + 1)) ** (self.var.lambda1[land_use_indices] / (self.var.lambda1[land_use_indices] + 1)))
            self.var.wwp2[land_use_indices] = self.var.wres2[land_use_indices] + (self.var.ws2[land_use_indices] - self.var.wres2[land_use_indices]) / ((1 + (alpha2[land_use_indices] * (10**4.2)) ** (self.var.lambda2[land_use_indices] + 1)) ** (self.var.lambda2[land_use_indices] / (self.var.lambda2[land_use_indices] + 1)))
            self.var.wwp3[land_use_indices] = self.var.wres3[land_use_indices] + (self.var.ws3[land_use_indices] - self.var.wres3[land_use_indices]) / ((1 + (alpha3[land_use_indices] * (10**4.2)) ** (self.var.lambda3[land_use_indices] + 1)) ** (self.var.lambda3[land_use_indices] / (self.var.lambda3[land_use_indices] + 1)))

            satTerm1FC = np.maximum(0., self.var.wfc1[land_use_indices] - self.var.wres1[land_use_indices]) / (self.var.ws1[land_use_indices] - self.var.wres1[land_use_indices])
            satTerm2FC = np.maximum(0., self.var.wfc2[land_use_indices] - self.var.wres2[land_use_indices]) / (self.var.ws2[land_use_indices] - self.var.wres2[land_use_indices])
            satTerm3FC = np.maximum(0., self.var.wfc3[land_use_indices] - self.var.wres3[land_use_indices]) / (self.var.ws3[land_use_indices] - self.var.wres3[land_use_indices])
            kUnSat1FC = self.var.KSat1[land_use_indices] * np.sqrt(satTerm1FC) * np.square(1 - (1 - satTerm1FC ** (1 / (self.var.lambda1[land_use_indices] / (self.var.lambda1[land_use_indices] + 1)))) ** (self.var.lambda1[land_use_indices] / (self.var.lambda1[land_use_indices] + 1)))
            kUnSat2FC = self.var.KSat2[land_use_indices] * np.sqrt(satTerm2FC) * np.square(1 - (1 - satTerm2FC ** (1 / (self.var.lambda2[land_use_indices] / (self.var.lambda2[land_use_indices] + 1)))) ** (self.var.lambda2[land_use_indices] / (self.var.lambda2[land_use_indices] + 1)))
            kUnSat3FC = self.var.KSat3[land_use_indices] * np.sqrt(satTerm3FC) * np.square(1 - (1 - satTerm3FC ** (1 / (self.var.lambda3[land_use_indices] / (self.var.lambda3[land_use_indices] + 1)))) ** (self.var.lambda3[land_use_indices] / (self.var.lambda3[land_use_indices] + 1)))
            self.var.kunSatFC12[land_use_indices] = np.sqrt(kUnSat1FC * kUnSat2FC)
            self.var.kunSatFC23[land_use_indices] = np.sqrt(kUnSat2FC * kUnSat3FC)

        # for paddy irrigation flooded paddy fields
        self.var.topwater = self.model.data.HRU.load_initial("topwater", default=self.var.full_compressed(0, dtype=np.float32))
        self.var.adjRoot = np.tile(self.var.full_compressed(np.nan, dtype=np.float32), (self.var.soilLayers, 1))

        self.var.arnoBeta = self.var.full_compressed(np.nan, dtype=np.float32)

        # Improved Arno's scheme parameters: Hageman and Gates 2003
        # arnoBeta defines the shape of soil water capacity distribution curve as a function of  topographic variability
        # b = max( (oh - o0)/(oh + omax), 0.01)
        # oh: the standard deviation of orography, o0: minimum std dev, omax: max std dev
        arnoBetaOro = (ElevationStD - 10.0) / (ElevationStD + 1500.0)

        # for CALIBRATION
        arnoBetaOro = arnoBetaOro + self.model.data.to_HRU(data=loadmap('arnoBeta_add'), fn=None)  # checked
        arnoBetaOro = np.minimum(1.2, np.maximum(0.01, arnoBetaOro))

        initial_humidy = 0.5
        self.var.w1 = self.model.data.HRU.load_initial('w1', default=np.nan_to_num(self.var.wwp1 + initial_humidy * (self.var.wfc1-self.var.wwp1)))
        self.var.w2 = self.model.data.HRU.load_initial('w2', default=np.nan_to_num(self.var.wwp2 + initial_humidy * (self.var.wfc2-self.var.wwp2)))
        self.var.w3 = self.model.data.HRU.load_initial('w3', default=np.nan_to_num(self.var.wwp3 + initial_humidy * (self.var.wfc3-self.var.wwp3)))

        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            # other paramater values
            # b coefficient of soil water storage capacity distribution
            #self.var.minCropKC.append(loadmap(coverType + "_minCropKC"))

            #self.var.minInterceptCap.append(loadmap(coverType + "_minInterceptCap"))
            #self.var.cropDeplFactor.append(loadmap(coverType + "_cropDeplFactor"))
            # parameter values

            land_use_indices = np.where(self.var.land_use_type == coverNum)[0]

            arnoBeta = self.model.data.to_HRU(data=loadmap(coverType + "_arnoBeta"), fn=None)
            if not isinstance(arnoBeta, float):
                arnoBeta = arnoBeta[land_use_indices]
            self.var.arnoBeta[land_use_indices] = (arnoBetaOro + arnoBeta)[land_use_indices]  # checked
            self.var.arnoBeta[land_use_indices] = np.minimum(1.2, np.maximum(0.01, self.var.arnoBeta[land_use_indices]))

            # Due to large rooting depths, the third (final) soil layer may be pushed to its minimum of 0.05 m.
            # In such a case, it may be better to turn off the root fractioning feature, as there is limited depth
            # in the third soil layer to hold water, while having a significant fraction of the rootss.
            # TODO: Extend soil depths to match maximum root depths
            
            rootFrac = np.tile(self.var.full_compressed(np.nan, dtype=np.float32), (self.var.soilLayers, 1))
            fractionroot12 = self.var.rootDepth1[land_use_indices] / (self.var.rootDepth1[land_use_indices] + self.var.rootDepth2[land_use_indices])
            rootFrac[0][land_use_indices] = fractionroot12 * rootFraction1[land_use_indices]
            rootFrac[1][land_use_indices] = (1 - fractionroot12) * rootFraction1[land_use_indices]
            rootFrac[2][land_use_indices] = 1.0 - rootFraction1[land_use_indices]

            if 'rootFrac' in binding:
                if not checkOption('rootFrac'):
                    root_depth_sum = self.var.rootDepth[0][land_use_indices] + self.var.rootDepth[1][land_use_indices] + self.var.rootDepth[2][land_use_indices]
                    for layer in range(3):
                        rootFrac[layer] = self.var.rootDepth[layer][land_use_indices] / root_depth_sum

            for soilLayer in range(self.var.soilLayers):
                self.var.adjRoot[soilLayer][land_use_indices] = rootFrac[soilLayer][land_use_indices] / np.sum(rootFrac, axis=0)[land_use_indices]

        # for maximum of topwater flooding (default = 0.05m)
        if "irrPaddy_maxtopwater" in binding:
            self.var.maxtopwater = loadmap('irrPaddy_maxtopwater')
        else:
            self.var.maxtopwater = 0.05

        # for irrigation of non paddy -> No =3
        print("look at this, really not multiply by root depth?")
        totalWaterPlant1 = np.maximum(0., self.var.wfc1 - self.var.wwp1) #* self.var.rootDepth[0][3]
        totalWaterPlant2 = np.maximum(0., self.var.wfc2 - self.var.wwp2) #* self.var.rootDepth[1][3]
        #totalWaterPlant3 = np.maximum(0., self.var.wfc3[3] - self.var.wwp3[3]) * self.var.rootDepth[2][3]  # Why is this turned off? MS
        self.var.totAvlWater = totalWaterPlant1 + totalWaterPlant2 #+ totalWaterPlant3

        # self.var.GWVolumeVariation = 0
        # self.var.ActualPumpingRate = 0
        self.forest_kc_ds = xr.open_dataset(self.model.model_structure['forcing']['landcover/forest/cropCoefficientForest_10days'])['cropCoefficientForest_10days']

    def water_body_exchange(self, groundwater_recharge):
        """computing leakage from rivers"""
        riverbedExchangeM3 = self.model.data.grid.leakageriver_factor * self.var.cellArea * ((1 - self.var.capriseindex + 0.25) // 1)
        riverbedExchangeM3[self.var.land_use_type != 5] = 0
        riverbedExchangeM3 = self.model.data.to_grid(HRU_data=riverbedExchangeM3, fn='sum')
        riverbedExchangeM3 = np.minimum(
            riverbedExchangeM3,
            0.80 * self.model.data.grid.channelStorageM3
        )
        # if there is a lake in this cell, there is no leakage
        riverbedExchangeM3[self.model.data.grid.waterBodyID > 0] = 0

        # adding leakage from river to the groundwater recharge
        waterbed_recharge = self.model.data.grid.M3toM(riverbedExchangeM3)
        
        # riverbed exchange means water is being removed from the river to recharge
        self.model.data.grid.riverbedExchangeM3 = riverbedExchangeM3  # to be used in routing_kinematic

        # first, lakes variable need to be extended to their area and not only to the discharge point
        lakeIDbyID = np.unique(self.model.data.grid.waterBodyID)

        lakestor_id = np.copy(self.model.data.grid.lakeStorage)
        resstor_id = np.copy(self.model.data.grid.resStorage)
        for id in range(len(lakeIDbyID)):  # for each lake or reservoir
            if lakeIDbyID[id] != 0:
                temp_map = np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id], np.where(self.model.data.grid.lakeStorage > 0, 1, 0), 0)  # Looking for the discharge point of the lake
                if np.sum(temp_map) == 0:  # try reservoir
                    temp_map = np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id], np.where(self.model.data.grid.resStorage > 0, 1, 0), 0)  # Looking for the discharge point of the reservoir
                discharge_point = np.nanargmax(temp_map)  # Index of the cell where the lake outlet is stored
                if self.model.data.grid.waterBodyTypTemp[discharge_point] != 0:

                    if self.model.data.grid.waterBodyTypTemp[discharge_point] == 1:  # this is a lake
                        # computing the lake area
                        area_stor = np.sum(np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id], self.model.data.grid.cellArea, 0))  # required to keep mass balance rigth
                        # computing the lake storage in meter and put this value in each cell including the lake
                        lakestor_id = np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id],
                                                self.model.data.grid.lakeStorage[discharge_point] / area_stor, lakestor_id)  # in meter

                    else:  # this is a reservoir
                        # computing the reservoir area
                        area_stor = np.sum(np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id], self.model.data.grid.cellArea, 0))  # required to keep mass balance rigth
                        # computing the reservoir storage in meter and put this value in each cell including the reservoir
                        resstor_id = np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id],
                                                self.model.data.grid.resStorage[discharge_point] / area_stor, resstor_id)  # in meter

        # Gathering lakes and reservoirs in the same array
        lakeResStorage = np.where(self.model.data.grid.waterBodyTypTemp == 0, 0., np.where(self.model.data.grid.waterBodyTypTemp == 1,
                                                                                lakestor_id, resstor_id))  # in meter

        minlake = np.maximum(0., 0.98 * lakeResStorage)  # reasonable but arbitrary limit

        # leakage depends on water bodies storage, water bodies fraction and modflow saturated area
        lakebedExchangeM = self.model.data.grid.leakagelake_factor * ((1 - self.var.capriseindex + 0.25) // 1)
        lakebedExchangeM[self.var.land_use_type != 5] = 0
        lakebedExchangeM = self.model.data.to_grid(HRU_data=lakebedExchangeM, fn='sum')
        lakebedExchangeM = np.minimum(
            lakebedExchangeM,
            minlake
        )

        # Now, leakage is converted again from the lake/reservoir area to discharge point to be removed from the lake/reservoir store
        self.model.data.grid.lakebedExchangeM3 = np.zeros(self.model.data.grid.compressed_size, dtype=np.float32)
        for id in range(len(lakeIDbyID)):  # for each lake or reservoir
            if lakeIDbyID[id] != 0:
                temp_map = np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id], np.where(self.model.data.grid.lakeStorage > 0, 1, 0), 0)  # Looking for the discharge point of the lake
                if np.sum(temp_map) == 0:  # try reservoir
                    temp_map = np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id], np.where(self.model.data.grid.resStorage > 0, 1, 0), 0)  # Looking for the discharge point of the reservoir
                discharge_point = np.nanargmax(temp_map)  # Index of the cell where the lake outlet is stored
            # Converting the lake/reservoir leakage from meter to cubic meter and put this value in the cell corresponding to the outlet
            self.model.data.grid.lakebedExchangeM3[discharge_point] = np.sum(np.where(self.model.data.grid.waterBodyID == lakeIDbyID[id],
                                                                                                lakebedExchangeM * self.model.data.grid.cellArea, 0))  # in m3
        self.model.data.grid.lakebedExchangeM = self.model.data.grid.M3toM(self.model.data.grid.lakebedExchangeM3)

        # compressed version for lakes and reservoirs
        lakeExchangeM3 = np.compress(self.model.data.grid.compress_LR, self.model.data.grid.lakebedExchangeM) * self.model.data.grid.MtoM3C

        # substract from both, because it is sorted by self.var.waterBodyTypCTemp
        self.model.data.grid.lakeStorageC = self.model.data.grid.lakeStorageC - lakeExchangeM3
        # assert (self.model.data.grid.lakeStorageC >= 0).all()
        self.model.data.grid.lakeVolumeM3C = self.model.data.grid.lakeVolumeM3C - lakeExchangeM3
        self.model.data.grid.reservoirStorageM3C = self.model.data.grid.reservoirStorageM3C - lakeExchangeM3

        # and from the combined one for waterbalance issues
        self.model.data.grid.lakeResStorageC = self.model.data.grid.lakeResStorageC - lakeExchangeM3
        # assert (self.model.data.grid.lakeResStorageC >= 0).all()
        self.model.data.grid.lakeResStorage = self.model.data.grid.full_compressed(0, dtype=np.float32)
        np.put(self.model.data.grid.lakeResStorage, self.model.data.grid.decompress_LR, self.model.data.grid.lakeResStorageC)

        # adding leakage from lakes and reservoirs to the groundwater recharge
        waterbed_recharge += lakebedExchangeM

        groundwater_recharge += waterbed_recharge

    def dynamic(self):
        """
        Dynamic part of the land cover type module

        Calculating soil for each of the 6  land cover class

        * calls evaporation_module.dynamic
        * calls interception_module.dynamic
        * calls soil_module.dynamic
        * calls sealed_water_module.dynamic

        And sums every thing up depending on the land cover type fraction
        """

        if checkOption('calcWaterBalance'):
            interceptStor_pre = self.var.interceptStor.copy()
            w1_pre = self.var.w1.copy()
            w2_pre = self.var.w2.copy()
            w3_pre = self.var.w3.copy()
            topwater_pre = self.var.topwater.copy()

        crop_stage_lenghts = np.column_stack([
            self.farmers.crop_variables['d1'],
            self.farmers.crop_variables['d2a'] + self.farmers.crop_variables['d2b'],
            self.farmers.crop_variables['d3a'] + self.farmers.crop_variables['d3b'],
            self.farmers.crop_variables['d4']
        ])

        crop_factors = np.column_stack([
            self.farmers.crop_variables['Kc1'],
            self.farmers.crop_variables['Kc3'],
            self.farmers.crop_variables['Kc5'],
        ])

        self.var.cropKC = get_crop_kc(
            self.var.crop_map,
            self.var.crop_age_days_map,
            self.var.crop_harvest_age_days,
            crop_stage_lenghts,
            crop_factors
        )
        if self.model.use_gpu:
            self.var.cropKC = cp.array(self.var.cropKC)



        forest_cropCoefficientNC = self.model.data.to_HRU(
            data=self.model.data.grid.compress(
                self.forest_kc_ds.sel(
                    time=self.model.current_time.replace(year=2000),
                    method='ffill').data
            ),
            fn=None
        )

        self.grass_kc_ds = xr.open_dataset(self.model.model_structure['forcing']['landcover/grassland/cropCoefficientGrassland_10days'])['cropCoefficientGrassland_10days']
        grassland_cropCoefficientNC = self.model.data.to_HRU(
            data=self.model.data.grid.compress(
                self.grass_kc_ds.sel(
                    time=self.model.current_time.replace(year=2000),
                    method='ffill').data
            ),
            fn=None
        )
        
        self.var.cropKC[self.var.land_use_type == 0] = forest_cropCoefficientNC[self.var.land_use_type == 0]
        self.var.cropKC[self.var.land_use_type == 1] = grassland_cropCoefficientNC[self.var.land_use_type == 1]

        potTranspiration, potBareSoilEvap, totalPotET, self.potET_forest,self.potET_grassland,self.potET_agriculture, self.cropkc_forest, self.cropkc_grassland, self.cropkc_agriculture = self.model.evaporation_module.dynamic(self.var.ETRef)
        
        if self.model.config['general']['simulate_forest']:
            print('check whether this is correct with plantFATE implementation')
        potTranspiration, self.interceptcap_forest, self.interceptcap_grassland,  self.interceptcap_agriculture, self.interceptevap_forest, self.interceptevap_grassland,  self.interceptevap_agriculture,  self.rain_forest,  self.rain_agriculture,  self.rain_grassland = self.model.interception_module.dynamic(potTranspiration)  # first thing that evaporates is the water intercepted water.

        # *********  WATER Demand   *************************
        groundwater_abstaction, channel_abstraction_m, addtoevapotrans, returnFlow = self.model.waterdemand_module.dynamic(totalPotET)

        openWaterEvap = self.var.full_compressed(0, dtype=np.float32)
        # Soil for forest, grassland, and irrigated land
        capillar = self.model.data.to_HRU(data=self.model.data.grid.capillar, fn=None)
        del self.model.data.grid.capillar

        interflow, directRunoff, groundwater_recharge, perc3toGW, prefFlow, openWaterEvap, self.et_forest, self.et_grassland, self.et_agriculture, self.soilwaterstorage_forest,self.soilwaterstorage_grassland, self.soilwaterstorage_agriculture, self.infiltration_forest, self.infiltration_grassland, self.infiltration_agriculture, self.potentialinfiltration_forest, self.potentialinfiltration_grassland, self.potentialinfiltration_agriculture, self.transpiration_decid, self.transpiration_conifer, self.transpiration_mixed, self.percolation_forest,self.percolation_agriculture,self.percolation_grassland,self.baresoil_forest,self.baresoil_agriculture, self.baresoil_grassland, self.soilwaterstorage_relsat_forest, self.soilwaterstorage_relsat_grassland, self.soilwaterstorage_relsat_agriculture, self.soilwaterstorage_full= self.model.soil_module.dynamic(
            capillar,
            openWaterEvap,
            potTranspiration,
            potBareSoilEvap,
            totalPotET
        )
        directRunoff = self.model.sealed_water_module.dynamic(capillar, openWaterEvap, directRunoff)


        #directRunoff = self.model.runoff_concentration_module.dynamic(interflow, directRunoff)
        land_use_indices_forest = np.where(self.var.land_use_type == 0)
        land_use_indices_grass_agri = np.where((self.var.land_use_type == 1) | (self.var.land_use_type == 3))
        bioarea = np.where(self.var.land_use_type < 4)[0].astype(np.int32)

        if self.model.current_timestep == 1:
            self.var.runoff_delay = self.var.full_compressed(0, dtype=np.float32)
            self.var.runoff_delay[land_use_indices_grass_agri] = directRunoff[land_use_indices_grass_agri] * 0.02
            self.var.runoff_delay[land_use_indices_forest] = directRunoff[land_use_indices_forest] * 0.05
            self.var.runoff_delay_pre = self.var.runoff_delay.copy()
            directRunoff[bioarea] = directRunoff[bioarea] - self.var.runoff_delay[bioarea]
        else:
            self.var.runoff_delay_pre = self.var.runoff_delay.copy()
            directRunoff[land_use_indices_forest] = directRunoff[land_use_indices_forest] + self.var.runoff_delay_pre[land_use_indices_forest]
            self.var.runoff_delay[land_use_indices_grass_agri] = directRunoff[land_use_indices_grass_agri] * 0.02
            self.var.runoff_delay[land_use_indices_forest] = directRunoff[land_use_indices_forest] * 0.05
            directRunoff[bioarea] = directRunoff[bioarea] - self.var.runoff_delay[bioarea]

        self.directrunoff_forest= self.var.full_compressed(0, dtype=np.float32)
        self.directrunoff_agriculture = self.var.full_compressed(0, dtype=np.float32)
        self.directrunoff_grassland = self.var.full_compressed(0, dtype=np.float32)
        self.interflow_forest = self.var.full_compressed(0, dtype=np.float32)
        self.interflow_agriculture = self.var.full_compressed(0, dtype=np.float32)
        self.interflow_grassland = self.var.full_compressed(0, dtype=np.float32)

        self.directrunoff_forest[:] = sum(directRunoff[self.var.land_use_indices_forest] *self.var.area_forest_ref)
        self.directrunoff_agriculture[:] = sum(directRunoff[self.var.land_use_indices_agriculture] *self.var.area_agriculture_ref)
        self.directrunoff_grassland[:] = sum(directRunoff[self.var.land_use_indices_grassland] *self.var.area_grassland_ref)
        self.interflow_forest[:] = sum(interflow[self.var.land_use_indices_forest]*self.var.area_forest_ref)
        self.interflow_agriculture[:] = sum(interflow[self.var.land_use_indices_agriculture]*self.var.area_agriculture_ref)
        self.interflow_grassland[:] = sum(interflow[self.var.land_use_indices_grassland] *self.var.area_grassland_ref)


        if self.model.use_gpu:
            self.var.actual_transpiration_crop[self.var.crop_map != -1] += self.var.actTransTotal.get()[self.var.crop_map != -1]
            self.var.potential_transpiration_crop[self.var.crop_map != -1] += potTranspiration.get()[self.var.crop_map != -1]
        else:
            self.var.actual_transpiration_crop[self.var.crop_map != -1] += self.var.actTransTotal[self.var.crop_map != -1]
            self.var.potential_transpiration_crop[self.var.crop_map != -1] += potTranspiration[self.var.crop_map != -1]

        assert not np.isnan(interflow).any()
        assert not np.isnan(groundwater_recharge).any()
        assert not np.isnan(groundwater_abstaction).any()
        assert not np.isnan(channel_abstraction_m).any()
        assert not np.isnan(openWaterEvap).any()

        if checkOption('calcWaterBalance'):
            self.model.waterbalance_module.waterBalanceCheck(
                how='cellwise',
                influxes=[self.var.Rain, self.var.SnowMelt],
                outfluxes=[self.var.natural_available_water_infiltration, self.var.interceptEvap],
                prestorages=[interceptStor_pre],
                poststorages=[self.var.interceptStor],
                tollerance=1e-6
            )

            self.model.waterbalance_module.waterBalanceCheck(
                how='cellwise',
                influxes=[self.var.natural_available_water_infiltration, capillar, self.var.actual_irrigation_consumption],
                outfluxes=[
                    directRunoff, perc3toGW, prefFlow,
                    self.var.actTransTotal, self.var.actBareSoilEvap, openWaterEvap
                ],
                prestorages=[w1_pre, w2_pre, w3_pre, topwater_pre],
                poststorages=[self.var.w1, self.var.w2, self.var.w3, self.var.topwater],
                tollerance=1e-6
            )

            totalstorage = np.sum(self.var.SnowCoverS, axis=0) / self.var.numberSnowLayersFloat + self.var.interceptStor + self.var.w1 + self.var.w2 + self.var.w3 + self.var.topwater
            totalstorage_pre = self.var.prevSnowCover + w1_pre + w2_pre + w3_pre + topwater_pre + interceptStor_pre

            self.model.waterbalance_module.waterBalanceCheck(
                how='cellwise',
                influxes=[self.var.precipitation_m_day, self.var.actual_irrigation_consumption, capillar],
                outfluxes=[
                    directRunoff, interflow, groundwater_recharge,
                    self.var.actTransTotal, self.var.actBareSoilEvap, openWaterEvap,
                    self.var.interceptEvap, self.var.snowEvap
                ],
                prestorages=[totalstorage_pre],
                poststorages=[totalstorage],
                tollerance=1e-6
            )

        groundwater_recharge = self.model.data.to_grid(HRU_data=groundwater_recharge, fn='mean')
        if checkOption('usewaterbodyexchange'):
            self.water_body_exchange(groundwater_recharge)
        else:
            self.model.data.grid.riverbedExchangeM3 = 0

        return (
            self.model.data.to_grid(HRU_data=interflow,fn='mean'),
            self.model.data.to_grid(HRU_data=directRunoff,fn='mean'),
            groundwater_recharge, 
            groundwater_abstaction, 
            channel_abstraction_m, 
            openWaterEvap, 
            returnFlow,
            self.potET_forest,self.potET_grassland,self.potET_agriculture, self.cropkc_forest, self.cropkc_grassland, self.cropkc_agriculture,
            self.interceptcap_forest, self.interceptcap_grassland,  self.interceptcap_agriculture, self.interceptevap_forest, self.interceptevap_grassland,  self.interceptevap_agriculture,  self.rain_forest,  self.rain_agriculture,  self.rain_grassland,
            self.et_forest, self.et_grassland, self.et_agriculture, self.soilwaterstorage_forest,self.soilwaterstorage_grassland, self.soilwaterstorage_agriculture, self.infiltration_forest, self.infiltration_grassland, self.infiltration_agriculture, self.potentialinfiltration_forest, self.potentialinfiltration_grassland, self.potentialinfiltration_agriculture, self.transpiration_decid, self.transpiration_conifer, self.transpiration_mixed, self.percolation_forest,self.percolation_agriculture,self.percolation_grassland,self.baresoil_forest,self.baresoil_agriculture, self.baresoil_grassland, self.interflow_forest, self.interflow_agriculture, self.interflow_grassland, self.directrunoff_forest, self.directrunoff_agriculture, self.directrunoff_grassland, self.soilwaterstorage_relsat_forest, self.soilwaterstorage_relsat_grassland, self.soilwaterstorage_relsat_agriculture, self.soilwaterstorage_full


        )