# -------------------------------------------------------------------------
# Name:        Land Cover Type module
# Purpose:
#
# Author:      PB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------
from osgeo import gdal
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    pass
import pandas as pd
from numba import njit
import calendar
from datetime import datetime
from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import checkOption, readnetcdf2, returnBool, cbinding, binding, loadmap, divideValues


@njit
def interpolate_kc(stage_start, stage_end, crop_progress, stage_start_kc, stage_end_kc):
    stage_progress = (crop_progress - stage_start) / (stage_end - stage_start)
    return (stage_end_kc - stage_start_kc) * stage_progress + stage_start_kc

@njit
def get_crop_kc(crop_map, crop_age_days_map, crop_harvest_age_days_map, crop_stage_data, kc_crop_stage):
    shape = crop_map.shape
    crop_map = crop_map.ravel()
    crop_age_days_map = crop_age_days_map.ravel()
    crop_harvest_age_days_map = crop_harvest_age_days_map.ravel()
    
    kc = np.full(crop_map.size, np.nan, dtype=np.float32)

    for i in range(crop_map.size):
        crop = crop_map[i]
        if crop != -1:
            age_days = crop_age_days_map[i]
            harvest_day = crop_harvest_age_days_map[i]
            crop_progress = age_days / harvest_day
            stage = np.searchsorted(crop_stage_data[crop], crop_progress, side='left')
            if stage == 0:
                field_kc = kc_crop_stage[crop, 0]
            elif stage == 1:
                field_kc = interpolate_kc(
                    stage_start=crop_stage_data[crop, 0],
                    stage_end=crop_stage_data[crop, 1],
                    crop_progress=crop_progress,
                    stage_start_kc=kc_crop_stage[crop, 0],
                    stage_end_kc=kc_crop_stage[crop, 1]
                )
            elif stage == 2:
                field_kc = kc_crop_stage[crop, 1]
            else:
                assert stage == 3
                field_kc = interpolate_kc(
                    stage_start=crop_stage_data[crop, 2],
                    stage_end=1,
                    crop_progress=crop_progress,
                    stage_start_kc=kc_crop_stage[crop, 1],
                    stage_end_kc=kc_crop_stage[crop, 2]
                )
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
        self.var = model.data.landunit
        self.model = model
        self.farmers = model.agents.farmers

    def initial(self, ElevationStD, soildepth):
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

        self.var.capriseindex = self.var.full_compressed(0, dtype=np.float32)

        self.var.actBareSoilEvap = self.var.full_compressed(0, dtype=np.float32)
        self.var.actTransTotal = self.var.full_compressed(0, dtype=np.float32)

        self.var.minCropKC = loadmap('minCropKC')

        rootFraction1 = self.var.full_compressed(np.nan, dtype=np.float32)
        maxRootDepth = self.var.full_compressed(np.nan, dtype=np.float32)
        soildepth_factor = loadmap('soildepth_factor')
        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            land_use_indices = np.where(self.var.land_use_type == coverNum)
            rootFraction1[land_use_indices] = self.model.data.to_landunit(data=loadmap(coverType + "_rootFraction1"), fn=None)[land_use_indices]
            maxRootDepth[land_use_indices] = self.model.data.to_landunit(data=loadmap(coverType + "_maxRootDepth") * soildepth_factor, fn=None)[land_use_indices]

        rootDepth1 = self.var.full_compressed(np.nan, dtype=np.float32)
        rootDepth2 = self.var.full_compressed(np.nan, dtype=np.float32)
        rootDepth3 = self.var.full_compressed(np.nan, dtype=np.float32)
        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            land_use_indices = np.where(self.var.land_use_type == coverNum)
            # calculate rootdepth for each soillayer and each land cover class
            rootDepth1[land_use_indices] = soildepth[0][land_use_indices]  # 0.05 m
            if coverNum in (0, 2, 3):  # forest, paddy irrigated, non-paddy irrigated
                # soil layer 1 = root max of land cover - first soil layer
                h1 = np.maximum(soildepth[1][land_use_indices], maxRootDepth[land_use_indices] - soildepth[0][land_use_indices])
                #
                rootDepth2[land_use_indices] = np.minimum(soildepth[1][land_use_indices] + soildepth[2][land_use_indices] - 0.05, h1)
                # soil layer is minimum 0.05 m
                rootDepth3[land_use_indices] = np.maximum(0.05, soildepth[1][land_use_indices] + soildepth[2][land_use_indices] - rootDepth2[land_use_indices]) # What is the motivation of this pushing the roodDepth[1] to the maxRotDepth
            else:
                assert coverNum == 1  # grassland
                rootDepth2[land_use_indices] = soildepth[1][land_use_indices]
                rootDepth3[land_use_indices] = soildepth[2][land_use_indices]

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
            land_use_indices = np.where(self.var.land_use_type == coverNum)
            # for forest there is a special map, for the other land use types the same map is used
            if coverType == 'forest':
                pre = "forest_"
            else:
                pre = ""
            # ksat in cm/d-1 -> m/dm
            self.var.KSat1[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "KSat1")/100, fn=None)[land_use_indices]  # checked
            self.var.KSat2[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "KSat2")/100, fn=None)[land_use_indices]  # checked
            self.var.KSat3[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "KSat3")/100, fn=None)[land_use_indices]  # checked
            alpha1[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "alpha1"), fn=None)[land_use_indices]  # checked
            alpha2[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "alpha2"), fn=None)[land_use_indices]  # checked
            alpha3[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "alpha3"), fn=None)[land_use_indices]  # checked
            self.var.lambda1[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "lambda1"), fn=None)[land_use_indices]  # checked
            self.var.lambda2[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "lambda2"), fn=None)[land_use_indices]  # checked
            self.var.lambda3[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "lambda3"), fn=None)[land_use_indices]  # checked
            thetas1[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "thetas1"), fn=None)[land_use_indices]  # checked
            thetas2[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "thetas2"), fn=None)[land_use_indices]  # checked
            thetas3[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "thetas3"), fn=None)[land_use_indices]  # checked
            thetar1[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "thetar1"), fn=None)[land_use_indices]  # checked
            thetar2[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "thetar2"), fn=None)[land_use_indices]  # checked
            thetar3[land_use_indices] = self.model.data.to_landunit(data=loadmap(pre + "thetar3"), fn=None)[land_use_indices]  # checked
            
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
            self.var.ws1[land_use_indices] = thetas1[land_use_indices] * rootDepth1[land_use_indices]
            self.var.ws2[land_use_indices] = thetas2[land_use_indices] * rootDepth2[land_use_indices]
            self.var.ws3[land_use_indices] = thetas3[land_use_indices] * rootDepth3[land_use_indices]

            self.var.wres1[land_use_indices] = thetar1[land_use_indices] * rootDepth1[land_use_indices]
            self.var.wres2[land_use_indices] = thetar2[land_use_indices] * rootDepth2[land_use_indices]
            self.var.wres3[land_use_indices] = thetar3[land_use_indices] * rootDepth3[land_use_indices]

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
        self.var.topwater = self.model.data.landunit.load_initial("topwater", default=self.var.full_compressed(0, dtype=np.float32))
        self.var.adjRoot = np.tile(self.var.full_compressed(np.nan, dtype=np.float32), (self.var.soilLayers, 1))

        self.var.arnoBeta = self.var.full_compressed(np.nan, dtype=np.float32)

        # Improved Arno's scheme parameters: Hageman and Gates 2003
        # arnoBeta defines the shape of soil water capacity distribution curve as a function of  topographic variability
        # b = max( (oh - o0)/(oh + omax), 0.01)
        # oh: the standard deviation of orography, o0: minimum std dev, omax: max std dev
        arnoBetaOro = (ElevationStD - 10.0) / (ElevationStD + 1500.0)

        # for CALIBRATION
        arnoBetaOro = arnoBetaOro + self.model.data.to_landunit(data=loadmap('arnoBeta_add'), fn=None)  # checked
        arnoBetaOro = np.minimum(1.2, np.maximum(0.01, arnoBetaOro))

        initial_humidy = 0.5
        self.var.w1 = self.model.data.landunit.load_initial('w1', default=np.nan_to_num(self.var.wwp1 + initial_humidy * (self.var.wfc1-self.var.wwp1)))
        self.var.w2 = self.model.data.landunit.load_initial('w2', default=np.nan_to_num(self.var.wwp2 + initial_humidy * (self.var.wfc2-self.var.wwp2)))
        self.var.w3 = self.model.data.landunit.load_initial('w3', default=np.nan_to_num(self.var.wwp3 + initial_humidy * (self.var.wfc3-self.var.wwp3)))

        for coverNum, coverType in enumerate(self.model.coverTypes[:4]):
            # other paramater values
            # b coefficient of soil water storage capacity distribution
            #self.var.minCropKC.append(loadmap(coverType + "_minCropKC"))

            #self.var.minInterceptCap.append(loadmap(coverType + "_minInterceptCap"))
            #self.var.cropDeplFactor.append(loadmap(coverType + "_cropDeplFactor"))
            # parameter values

            land_use_indices = np.where(self.var.land_use_type == coverNum)[0]

            arnoBeta = self.model.data.to_landunit(data=loadmap(coverType + "_arnoBeta"), fn=None)
            if not isinstance(arnoBeta, float):
                arnoBeta = arnoBeta[land_use_indices]
            self.var.arnoBeta[land_use_indices] = (arnoBetaOro + arnoBeta)[land_use_indices]  # checked
            self.var.arnoBeta[land_use_indices] = np.minimum(1.2, np.maximum(0.01, self.var.arnoBeta[land_use_indices]))

            # Due to large rooting depths, the third (final) soil layer may be pushed to its minimum of 0.05 m.
            # In such a case, it may be better to turn off the root fractioning feature, as there is limited depth
            # in the third soil layer to hold water, while having a significant fraction of the rootss.
            # TODO: Extend soil depths to match maximum root depths
            
            rootFrac = np.tile(self.var.full_compressed(np.nan, dtype=np.float32), (self.var.soilLayers, 1))
            fractionroot12 = rootDepth1[land_use_indices] / (rootDepth1[land_use_indices] + rootDepth2[land_use_indices])
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
        totalWaterPlant1 = np.maximum(0., self.var.wfc1 - self.var.wwp1) #* self.var.rootDepth[0][3]
        totalWaterPlant2 = np.maximum(0., self.var.wfc2 - self.var.wwp2) #* self.var.rootDepth[1][3]
        #totalWaterPlant3 = np.maximum(0., self.var.wfc3[3] - self.var.wwp3[3]) * self.var.rootDepth[2][3]  # Why is this turned off? MS
        self.var.totAvlWater = totalWaterPlant1 + totalWaterPlant2 #+ totalWaterPlant3

        # self.var.GWVolumeVariation = 0
        # self.var.ActualPumpingRate = 0

        crop_factors = self.farmers.get_crop_factors()
        
        self.crop_stage_data = np.zeros((26, 4), dtype=np.float32)
        self.crop_stage_data[:, 0] = crop_factors['L_ini']
        self.crop_stage_data[:, 1] = crop_factors['L_dev']
        self.crop_stage_data[:, 2] = crop_factors['L_mid']
        self.crop_stage_data[:, 3] = crop_factors['L_late']

        self.kc_crop_stage = np.zeros((26, 3), dtype=np.float32)
        self.kc_crop_stage[:, 0] = crop_factors['kc_ini']
        self.kc_crop_stage[:, 1] = crop_factors['kc_mid']
        self.kc_crop_stage[:, 2] = crop_factors['kc_end']

    def water_body_exchange(self, groundwater_recharge):
        """computing leakage from rivers"""
        riverbedExchangeM3 = self.model.data.grid.leakageriver_factor * self.var.cellArea * ((1 - self.var.capriseindex + 0.25) // 1)
        riverbedExchangeM3[self.var.land_use_type != 5] = 0
        riverbedExchangeM3 = self.model.data.to_grid(landunit_data=riverbedExchangeM3, fn='sum')
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
        lakebedExchangeM = self.model.data.to_grid(landunit_data=lakebedExchangeM, fn='sum')
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
        self.model.data.grid.lakeVolumeM3C = self.model.data.grid.lakeVolumeM3C - lakeExchangeM3
        self.model.data.grid.reservoirStorageM3C = self.model.data.grid.reservoirStorageM3C - lakeExchangeM3

        # and from the combined one for waterbalance issues
        self.model.data.grid.lakeResStorageC = self.model.data.grid.lakeResStorageC - lakeExchangeM3
        self.model.data.grid.lakeResStorage = self.model.data.grid.full_compressed(0, dtype=np.float32)
        np.put(self.model.data.grid.lakeResStorage, self.model.data.grid.decompress_LR, self.model.data.grid.lakeResStorageC)

        # adding leakage from lakes and reservoirs to the groundwater recharge
        waterbed_recharge += lakebedExchangeM

        groundwater_recharge += waterbed_recharge

    def dynamic(self, ETRef):
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


        self.var.cropKC = get_crop_kc(
            self.var.crop_map.get() if self.model.args.use_gpu else self.var.crop_map,
            self.var.crop_age_days_map.get() if self.model.args.use_gpu else self.var.crop_age_days_map,
            self.var.crop_harvest_age_days_map.get() if self.model.args.use_gpu else self.var.crop_harvest_age_days_map,
            self.crop_stage_data,
            self.kc_crop_stage
        )
        if self.model.args.use_gpu:
            self.var.cropKC = cp.array(self.var.cropKC)

        cover_cropCoefficientNC = self.model.data.to_landunit(
            data=readnetcdf2('forest_cropCoefficientNC', globals.dateVar['10day'], "10day"),
            fn=None
        )
        
        self.var.cropKC[self.var.land_use_type == 0] = cover_cropCoefficientNC[self.var.land_use_type == 0]
        self.var.cropKC[self.var.land_use_type == 1] = self.var.minCropKC

        potTranspiration, potBareSoilEvap, totalPotET = self.model.evaporation_module.dynamic(ETRef)
        potTranspiration = self.model.interception_module.dynamic(potTranspiration)

        # *********  WATER Demand   *************************
        groundwater_abstaction, channel_abstraction_m, addtoevapotrans, returnFlow = self.model.waterdemand_module.dynamic(totalPotET)

        openWaterEvap = self.var.full_compressed(0, dtype=np.float32)
        # Soil for forest, grassland, and irrigated land
        capillar = self.model.data.to_landunit(data=self.model.data.grid.capillar, fn=None)
        del self.model.data.grid.capillar

        interflow, directRunoff, groundwater_recharge, perc3toGW, prefFlow, openWaterEvap = self.model.soil_module.dynamic(capillar, openWaterEvap, potTranspiration, potBareSoilEvap, totalPotET)
        directRunoff = self.model.sealed_water_module.dynamic(capillar, openWaterEvap, directRunoff)

        self.farmers.actual_transpiration_crop += self.var.actTransTotal
        self.farmers.potential_transpiration_crop += potTranspiration

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
                influxes=[self.var.Precipitation, self.var.actual_irrigation_consumption, capillar],
                outfluxes=[
                    directRunoff, interflow, groundwater_recharge,
                    self.var.actTransTotal, self.var.actBareSoilEvap, openWaterEvap,
                    self.var.interceptEvap, self.var.snowEvap
                ],
                prestorages=[totalstorage_pre],
                poststorages=[totalstorage],
                tollerance=1e-6
            )

        groundwater_recharge = self.model.data.to_grid(landunit_data=groundwater_recharge, fn='mean')
        if checkOption('usewaterbodyexchange'):
            self.water_body_exchange(groundwater_recharge)
        else:
            self.model.data.grid.riverbedExchangeM3 = 0

        return (
            self.model.data.to_grid(landunit_data=interflow,fn='mean'),
            self.model.data.to_grid(landunit_data=directRunoff,fn='mean'),
            groundwater_recharge, 
            groundwater_abstaction, 
            channel_abstraction_m, 
            openWaterEvap, 
            returnFlow
        )