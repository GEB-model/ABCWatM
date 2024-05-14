# -------------------------------------------------------------------------
# Name:        Interception module
# Purpose:
#
# Author:      PB
#
# Created:     01/08/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import loadmap, divideValues, checkOption, cbinding
import numpy as np
import xarray as xr
# -------------------------------------------------------------------------
# Name:        Interception module
# Purpose:
#
# Author:      PB
#
# Created:     01/08/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import loadmap, divideValues, checkOption, cbinding
import numpy as np
import xarray as xr
import rioxarray


class interception(object):
    """
    INTERCEPTION


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    EWRef                 potential evaporation rate from water surface                                     m        
    waterbalance_module                                                                                              
    interceptCap          interception capacity of vegetation                                               m        
    minInterceptCap       Maximum interception read from file for forest and grassland land cover           m        
    interceptStor         simulated vegetation interception storage                                         m        
    Rain                  Precipitation less snow                                                           m        
    availWaterInfiltrati  quantity of water reaching the soil after interception, more snowmelt             m        
    SnowMelt              total snow melt from all layers                                                   m        
    interceptEvap         simulated evaporation from water intercepted by vegetation                        m        
    potTranspiration      Potential transpiration (after removing of evaporation)                           m        
    actualET              simulated evapotranspiration from soil, flooded area and vegetation               m        
    snowEvap              total evaporation from snow for a snow layers                                     m        
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.HRU
        self.model = model

    def initial(self):
        self.var.minInterceptCap = self.var.full_compressed(np.nan, dtype=np.float32)
        self.var.interceptStor = self.var.full_compressed(np.nan, dtype=np.float32)

        self.var.interceptStor = self.model.data.HRU.load_initial("interceptStor", default=self.model.data.HRU.full_compressed(0, dtype=np.float32))

        for coverNum, coverType in enumerate(self.model.coverTypes):
            coverType_indices = np.where(self.var.land_use_type == coverNum)
            self.var.minInterceptCap[coverType_indices] = self.model.data.to_HRU(data=loadmap(coverType + "_minInterceptCap"), fn=None)
        
        assert not np.isnan(self.var.interceptStor).any()
        assert not np.isnan(self.var.minInterceptCap).any()

        self.interception_ds = {}
        for land_cover in ('forest', 'grassland'):
            self.interception_ds[land_cover] = xr.open_dataset(
                self.model.model_structure['forcing'][f'landcover/{land_cover}/interceptCap{land_cover.title()}_10days']
            )[f'interceptCap{land_cover.title()}_10days']

    def dynamic(self, potTranspiration):
        """
        Dynamic part of the interception module
        calculating interception for each land cover class

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        :return: interception evaporation, interception storage, reduced pot. transpiration

        """

        if checkOption('calcWaterBalance'):
            interceptStor_pre = self.var.interceptStor.copy()

        interceptCap = self.var.full_compressed(np.nan, dtype=np.float32)
        for coverNum, coverType in enumerate(self.model.coverTypes):
            coverType_indices = np.where(self.var.land_use_type == coverNum)
            if coverType in ('forest', 'grassland'):
                covertype_interceptCapNC = self.model.data.to_HRU(
                    data=self.model.data.grid.compress(
                        self.interception_ds[coverType].sel(
                            time=self.model.current_time.replace(year=2000),
                            method='ffill').data
                    ),
                    fn=None
                )
                interceptCap[coverType_indices] = covertype_interceptCapNC[coverType_indices]
            else:
                interceptCap[coverType_indices] = self.var.minInterceptCap[coverType_indices]
        
        assert not np.isnan(interceptCap).any()

        
        land_use_indices_forest = np.where(self.var.land_use_type == 0) 
        land_use_indices_grassland = np.where(self.var.land_use_type == 1) 
        land_use_indices_agriculture = np.where((self.var.land_use_type == 2) | (self.var.land_use_type == 3))

        if 0.0 in interceptCap[land_use_indices_forest]:
            # Change all zeros to 0.00022 (minimum value in forests in this dataset)
            interceptCap[land_use_indices_forest] = np.where(interceptCap[land_use_indices_forest] == 0, 0.00022, interceptCap[land_use_indices_forest])


        
        
        if self.model.current_timestep == 1:
            

            forest_types = rioxarray.open_rasterio("C:/Users/romij/GEB/GEB_models/meuse/models/meuse/base/input/landsurface/forest_types_mode_high_res.tif", masked=True)
                        # 30'' grid of forests, with True/False. Here initialized as random data
            

            # Define the forest type values
            forest_type_values = [6, 7, 8]

            # Initialize an empty array to hold filtered indices
            filtered_array = np.array([], dtype=int)

            # Iterate through each forest type
            for forest_type in forest_type_values:
                    # Create a boolean mask for the current forest type
                forest_mask_3d_boolean = (forest_types.values == forest_type)
                forest_mask = forest_mask_3d_boolean[0, :, :]


                # Get the indices of HRUs associated with the current forest type
                HRU_indices = self.var.decompress(
                    np.arange(self.model.data.HRU.land_use_type.size)
                )
                HRUs_to_forest = np.unique(HRU_indices[forest_mask])
                HRUs_to_forest = HRUs_to_forest[HRUs_to_forest != -1]

                # Filter HRUs based on land use type (assuming 0 represents forest)
                indices_for_forest = HRUs_to_forest[self.var.land_use_type[HRUs_to_forest] == 0]

                # Depending on the forest type, modify interceptCap
                if forest_type == 6:  # Deciduous
                    self.var.indicesDeciduous = indices_for_forest
                    # Create a mask indicating elements of land_use_indices_forest NOT present in indices_for_forest these can be used to filter land_use_indices_forest for mismatches between forest_types dataset and the original land use indices.
                    mask = np.isin(land_use_indices_forest, indices_for_forest)
                    # Filter land_use_indices_forest using the mask
                    filtered_array = land_use_indices_forest[0][~mask[0]]
                elif forest_type == 7:  # Coniferous
                    self.var.indicesConifer = indices_for_forest
                    mask = ~np.isin(filtered_array, indices_for_forest)
                    # Filter filtered_array using the mask
                    filtered_array = filtered_array[mask]
                elif forest_type == 8:  # Mixed
                    self.var.indicesMixed = indices_for_forest
                    mask = ~np.isin(filtered_array, indices_for_forest)
                    # Filter land_use_indices_forest using the mask
                    filtered_array = filtered_array[mask]
                    self.var.indicesMismatched = filtered_array

                    def random_forest_type(percentage_to_transfer):

                        # Calculate the number of indices to transfer
                        indices_to_transfer = int(total_indices * percentage_to_transfer / 100)

                        # Get random indices to transfer
                        random_forest = np.random.choice(self.var.indicesMismatched, size=indices_to_transfer, replace=False)
                        random_forest = np.sort(random_forest)
                        
                        #update mismatched indices by masking 
                        mask = ~np.isin( self.var.indicesMismatched, random_forest)
                        self.var.indicesMismatched = self.var.indicesMismatched[mask] 
                        return random_forest
                    
                    total_indices = self.var.indicesMismatched.size
                    deciduous_mismatch = random_forest_type(69.74) #percentage of forest to be deciduous in mismatched array, percentage is based on the percentage per forest in the Copernicus forest cover type dataset 2018
                    conifer_mismatch = random_forest_type(17.01)
                    mixed_mismatch = self.var.indicesMismatched

                    self.var.indicesDeciduous = np.concatenate((self.var.indicesDeciduous, deciduous_mismatch))
                    self.var.indicesConifer = np.concatenate((self.var.indicesConifer, conifer_mismatch))
                    self.var.indicesMixed = np.concatenate((self.var.indicesMixed, mixed_mismatch))

        interceptCap[self.var.indicesDeciduous] = interceptCap[self.var.indicesDeciduous] *1.5
        interceptCap[self.var.indicesConifer] = interceptCap[self.var.indicesConifer] *2.63
        interceptCap[self.var.indicesMixed] = interceptCap[self.var.indicesMixed] * 2
        interceptCap[land_use_indices_agriculture] = np.nanmean(interceptCap[land_use_indices_grassland])

        


       
       # Mask the values where rain is zero
        rain = self.var.Rain[land_use_indices_forest]
        masked_rain = np.where(rain != 0, rain, np.nan)
        ratio = interceptCap[land_use_indices_forest]/(masked_rain)*100
        ratio_clipped = np.where(np.isnan(ratio), np.nan, np.clip(ratio, 0, 100))
        self.interceptcap_forest =  interceptCap[land_use_indices_forest]


        rain = self.var.Rain[land_use_indices_agriculture]
        masked_rain = np.where(rain != 0, rain, np.nan)
        ratio = interceptCap[land_use_indices_agriculture]/(masked_rain)*100
        ratio_clipped = np.clip(ratio, 0, 100)
        self.interceptcap_agriculture =  interceptCap[land_use_indices_agriculture]

        rain = self.var.Rain[land_use_indices_grassland]
        masked_rain = np.where(rain != 0, rain, np.nan)
        ratio = interceptCap[land_use_indices_grassland]/(masked_rain)*100
        ratio_clipped = np.clip(ratio, 0, 100)
        self.interceptcap_grassland =  interceptCap[land_use_indices_grassland]
        # Rain instead Pr, because snow is substracted later
        # assuming that all interception storage is used in the other time step
        throughfall = np.maximum(0.0, self.var.Rain + self.var.interceptStor - interceptCap)

        # update interception storage after throughfall
        self.var.interceptStor = self.var.interceptStor + self.var.Rain - throughfall
        

        # availWaterInfiltration Available water for infiltration: throughfall + snow melt
        self.var.natural_available_water_infiltration = np.maximum(0.0, throughfall + self.var.SnowMelt)

        sealed_area = np.where(self.var.land_use_type == 4)
        water_area = np.where(self.var.land_use_type == 5)
        land_use_indices_grass_agri = np.where((self.var.land_use_type == 1) | (self.var.land_use_type == 3))  # 'forest', 'grassland', 'irrPaddy', 'irrNonPaddy'

        self.var.interceptEvap = self.var.full_compressed(np.nan, dtype=np.float32)
        # interceptEvap evaporation from intercepted water (based on potTranspiration)
        self.var.interceptEvap[self.var.indicesDeciduous] = np.minimum(
            self.var.interceptStor[self.var.indicesDeciduous],
            potTranspiration[self.var.indicesDeciduous]* 1.42* divideValues(self.var.interceptStor[self.var.indicesDeciduous], interceptCap[self.var.indicesDeciduous]) ** (2./3. )
        )
        self.var.interceptEvap[self.var.indicesConifer] = np.minimum(
            self.var.interceptStor[self.var.indicesConifer],
            potTranspiration[self.var.indicesConifer]* 1.85 * divideValues(self.var.interceptStor[self.var.indicesConifer], interceptCap[self.var.indicesConifer]) ** (2./3.)
        )
        self.var.interceptEvap[self.var.indicesMixed] = np.minimum(
            self.var.interceptStor[self.var.indicesMixed],
            potTranspiration[self.var.indicesMixed]* 1.75 * divideValues(self.var.interceptStor[self.var.indicesMixed], interceptCap[self.var.indicesMixed]) ** (2./3. )
        )
        self.var.interceptEvap[land_use_indices_grass_agri] = np.minimum(
            self.var.interceptStor[land_use_indices_grass_agri],
            potTranspiration[land_use_indices_grass_agri] * divideValues(self.var.interceptStor[land_use_indices_grass_agri], interceptCap[land_use_indices_grass_agri]) ** (2./3. )
        )

        self.var.interceptEvap[sealed_area] = np.maximum(
            np.minimum(self.var.interceptStor[sealed_area], self.var.EWRef[sealed_area]),
            self.var.full_compressed(0, dtype=np.float32)[sealed_area]
        )

        self.var.interceptEvap[water_area] = 0  # never interception for water
      
        # update interception storage and potTranspiration
        self.var.interceptStor = self.var.interceptStor - self.var.interceptEvap
         # Mask the values where rain is zero
           # Mask the values where rain is zero
        self.rain_forest = self.var.Rain[land_use_indices_forest]
        masked_rain = np.where(self.rain_forest != 0, self.rain_forest, np.nan)
        ratio = self.var.interceptEvap[land_use_indices_forest]/(masked_rain)*100
        ratio_clipped = np.where(np.isnan(ratio), np.nan, np.clip(ratio, 0, 100))
        self.interceptevap_forest = self.var.interceptEvap[land_use_indices_forest]


        self.rain_agriculture = self.var.Rain[land_use_indices_agriculture]
        masked_rain = np.where(self.rain_agriculture != 0, self.rain_agriculture, np.nan)
        ratio = self.var.interceptEvap[land_use_indices_agriculture]/(masked_rain)*100
        ratio_clipped = np.clip(ratio, 0, 100)
        self.interceptevap_agriculture = self.var.interceptEvap[land_use_indices_agriculture]

        self.rain_grassland = self.var.Rain[land_use_indices_grassland]
        masked_rain = np.where(self.rain_grassland!= 0, self.rain_grassland, np.nan)
        ratio = self.var.interceptEvap[land_use_indices_grassland]/(masked_rain)*100
        ratio_clipped = np.clip(ratio, 0, 100)
        self.interceptevap_grassland = self.var.interceptEvap[land_use_indices_grassland]

        potTranspiration = np.maximum(0, potTranspiration - self.var.interceptEvap)

        # update actual evaporation (after interceptEvap)
        # interceptEvap is the first flux in ET, soil evapo and transpiration are added later
        self.var.actualET = self.var.interceptEvap + self.var.snowEvap

        if checkOption('calcWaterBalance'):
            self.model.waterbalance_module.waterBalanceCheck(
                how='cellwise',
                influxes=[self.var.Rain, self.var.SnowMelt],  # In
                outfluxes=[self.var.natural_available_water_infiltration, self.var.interceptEvap],  # Out
                prestorages=[interceptStor_pre],  # prev storage
                poststorages=[self.var.interceptStor],
                tollerance=1e-7
            )

        # if self.model.use_gpu:
            # self.var.interceptEvap = self.var.interceptEvap.get()

        # deciduous, coniferous, mixed for forest, grassland, agriculture


        return potTranspiration
        #self.interceptcap_forest, self.interceptcap_grassland,  self.interceptcap_agriculture, self.interceptevap_forest, self.interceptevap_grassland,  self.interceptevap_agriculture,  self.rain_forest,  self.rain_agriculture,  self.rain_grassland
