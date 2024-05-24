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

        if 0.0 in interceptCap[self.var.land_use_indices_forest]:
            # Change all zeros to 0.000186 (mean minimum value in forests in this dataset)
            interceptCap[self.var.land_use_indices_forest] = np.where(interceptCap[self.var.land_use_indices_forest] == 0, 0.000186, interceptCap[self.var.land_use_indices_forest])

        #determine indices of forest types to apply different interception capacities and evaporation rates
        if self.model.current_timestep == 1:


            forest_types = rioxarray.open_rasterio("C:/Users/romij/GEB/GEB_models/meuse/models/meuse/base/input/landsurface/forest_types.tif", masked=True)
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
                    mask = np.isin(self.var.land_use_indices_forest, indices_for_forest)
                    filtered_array = self.var.land_use_indices_forest[0][~mask[0]]
                elif forest_type == 7:  # Coniferous
                    self.var.indicesConifer = indices_for_forest
                    mask = ~np.isin(filtered_array, indices_for_forest)
                    filtered_array = filtered_array[mask]
                elif forest_type == 8:  # Mixed
                    self.var.indicesMixed = indices_for_forest
                    mask = ~np.isin(filtered_array, indices_for_forest)
                    filtered_array = filtered_array[mask]
                    self.var.indicesMismatched = filtered_array
                    

                    # the forest types are randomly distributed across the mismatched indices, according to the percentage they occur in in the Meuse, through the following function:

                    def random_forest_type(percentage_to_transfer):
                        total_indices = self.var.indicesMismatched.size

                        # Calculate the number of indices to transfer
                        indices_to_transfer = int(total_indices * percentage_to_transfer / 100)

                        # Get random indices to transfer
                        random_forest = np.random.choice(self.var.indicesMismatched, size=indices_to_transfer, replace=False)
                        random_forest = np.sort(random_forest)

                        #update mismatched indices by masking 
                        mask = ~np.isin( self.var.indicesMismatched, random_forest)
                        self.var.indicesMismatched = self.var.indicesMismatched[mask] 
                        return random_forest

                    #deciduous_mismatch = random_forest_type(69.74) #percentage of forest to be deciduous in mismatched array, percentage is based on the percentage per forest in the Copernicus forest cover type dataset 2018
                    #conifer_mismatch = random_forest_type(17.01)

                    #the following function was only ones to to save the new HRUs to raster. 
                    #After, the new raster was used to update the forest types through the code above the function random_forest_type and the function random_forest_type and the function save_HRU_to_raster were not needed anymore
                    def save_HRU_to_raster(data):
                        import rasterio
                        from rasterio.transform import Affine
                        from rasterio.crs import CRS
                        transform = Affine.from_gdal(*self.model.data.HRU.gt)
                        # Define the CRS (EPSG code)
                        crs = CRS.from_epsg(4302)

                        # Output raster file path
                        output_raster = 'C:/Users/romij/GEB/GEB_models/meuse/models/meuse/base/input/landsurface/forest_types.tif'

                        data= self.var.decompress(
                        data
                        )

                        # Writing the data to a raster file
                        with rasterio.open(
                        output_raster,
                        'w',
                        driver='GTiff',
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,  # number of bands
                        dtype=data.dtype,
                        crs=crs,
                        transform=transform
                        ) as dst:
                            dst.write(data, 1)
                    #self.var.land_use_type[self.var.indicesDeciduous] = 6
                    #self.var.land_use_type[self.var.indicesConifer] = 7
                    #self.var.land_use_type[self.var.indicesMixed] = 8
                    #save_HRU_to_raster(self.var.land_use_type)

                    #mixed_mismatch = self.var.indicesMismatched
                    #self.var.indicesDeciduous = np.concatenate((self.var.indicesDeciduous, deciduous_mismatch))
                    #self.var.indicesConifer = np.concatenate((self.var.indicesConifer, conifer_mismatch))
                    #self.var.indicesMixed = np.concatenate((self.var.indicesMixed, mixed_mismatch))

        interceptCap[self.var.indicesDeciduous] = interceptCap[self.var.indicesDeciduous] *1.5
        interceptCap[self.var.indicesConifer] = interceptCap[self.var.indicesConifer] *2.63
        interceptCap[self.var.indicesMixed] = interceptCap[self.var.indicesMixed] * 2
        interceptCap[self.var.land_use_indices_agriculture] = np.nanmean(interceptCap[self.var.land_use_indices_grassland])

        # Rain instead Pr, because snow is substracted later
        # assuming that all interception storage is used in the other time step
        throughfall = np.maximum(0.0, self.var.Rain + self.var.interceptStor - interceptCap)

        # update interception storage after throughfall
        self.var.interceptStor = self.var.interceptStor + self.var.Rain - throughfall

        # availWaterInfiltration Available water for infiltration: throughfall + snow melt
        self.var.natural_available_water_infiltration = np.maximum(0.0, throughfall + self.var.SnowMelt)

        sealed_area = np.where(self.var.land_use_type == 4)
        water_area = np.where(self.var.land_use_type == 5)
        bio_area = np.where(self.var.land_use_type < 4)  # 'forest', 'grassland', 'irrPaddy', 'irrNonPaddy'
        land_use_indices_grass_agri = np.where((self.var.land_use_type == 1) | (self.var.land_use_type == 3))

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

        self.var.interceptEvap[bio_area] = np.minimum(
            self.var.interceptStor[bio_area],
            potTranspiration[bio_area] * divideValues(self.var.interceptStor[bio_area], interceptCap[bio_area]) ** (2./3.)
        )

        self.var.interceptEvap[sealed_area] = np.maximum(
            np.minimum(self.var.interceptStor[sealed_area], self.var.EWRef[sealed_area]),
            self.var.full_compressed(0, dtype=np.float32)[sealed_area]
        )

        self.var.interceptEvap[water_area] = 0  # never interception for water

        self.interceptcap_forest = self.var.full_compressed(np.nan, dtype=np.float32)
        self.interceptcap_grassland = self.var.full_compressed(np.nan, dtype=np.float32)
        self.interceptcap_agriculture = self.var.full_compressed(np.nan, dtype=np.float32)
        self.interceptevap_forest = self.var.full_compressed(np.nan, dtype=np.float32)
        self.interceptevap_grassland = self.var.full_compressed(np.nan, dtype=np.float32)
        self.interceptevap_agriculture = self.var.full_compressed(np.nan, dtype=np.float32)
        self.rain_forest = self.var.full_compressed(np.nan, dtype=np.float32)
        self.rain_agriculture = self.var.full_compressed(np.nan, dtype=np.float32)
        self.rain_grassland = self.var.full_compressed(np.nan, dtype=np.float32)




        self.interceptcap_forest[:] = sum(interceptCap[self.var.land_use_indices_forest] * self.var.area_forest_ref)
        self.interceptcap_grassland[:] = sum(interceptCap[self.var.land_use_indices_grassland] * self.var.area_grassland_ref)
        self.interceptcap_agriculture[:] = sum(interceptCap[self.var.land_use_indices_agriculture] * self.var.area_agriculture_ref)
        self.interceptevap_forest[:] = sum(self.var.interceptEvap[self.var.land_use_indices_forest] * self.var.area_forest_ref)
        self.interceptevap_grassland[:] = sum(self.var.interceptEvap[self.var.land_use_indices_grassland] * self.var.area_grassland_ref)
        self.interceptevap_agriculture[:] = sum(self.var.interceptEvap[self.var.land_use_indices_agriculture] * self.var.area_agriculture_ref)
        self.rain_forest[:] = sum(self.var.Rain[self.var.bioarea] * self.var.area_bioarea_ref)
        self.rain_agriculture[:] = sum(self.var.Rain[self.var.land_use_indices_agriculture] * self.var.area_agriculture_ref)
        self.rain_grassland[:]= sum(self.var.Rain[self.var.land_use_indices_grassland] * self.var.area_grassland_ref)





        # update interception storage and potTranspiration
        self.var.interceptStor = self.var.interceptStor - self.var.interceptEvap
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

        return potTranspiration, self.interceptcap_forest, self.interceptcap_grassland,  self.interceptcap_agriculture, self.interceptevap_forest, self.interceptevap_grassland,  self.interceptevap_agriculture,  self.rain_forest,  self.rain_agriculture,  self.rain_grassland