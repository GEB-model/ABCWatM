from math import isclose
import numpy as np
import os
from cwatm.management_modules.data_handling import globals, cbinding, loadmap, returnBool
from cwatm.hydrological_modules.groundwater_modflow.modflow_model import ModFlowSimulation
import rasterio
from cwatm.management_modules.data_handling import Flags

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class groundwater_modflow:
    def __init__(self, model):
        self.var = model.var
        self.model = model

    def get_corrected_modflow_cell_area(self):
        return np.bincount(
            self.indices['ModFlow_index'],
            weights=np.invert(self.var.mask.astype(np.bool)).ravel()[self.indices['CWatM_index']] * self.indices['area'],
            minlength=self.modflow.basin.size
        ).reshape(self.modflow.basin.shape)

    def get_corrected_cwatm_cell_area(self):
        return (self.var.cellArea_uncompressed.ravel() - np.bincount(
            self.indices['CWatM_index'],
            weights=np.invert(self.modflow.basin).ravel()[self.indices['ModFlow_index']] * self.indices['area'],
            minlength=self.var.mask.size
        )).reshape(self.var.mask.shape)

    def CWATM2modflow(self, variable, correct_boundary=False):
        if correct_boundary:
            modflow_cell_area = self.modflow_cell_area_corrected.ravel()
            area = self.indices['modflow_area']
        else:
            modflow_cell_area = self.modflow_cell_area.ravel()
            area = self.indices['area']
        variable[self.var.mask == 1] = 0
        assert not (np.isnan(variable).any())
        array = (np.bincount(
            self.indices['ModFlow_index'],
            variable.ravel()[self.indices['CWatM_index']] * area,
            minlength=self.domain['nrow'] * self.domain['ncol']
        ) / modflow_cell_area).reshape((self.domain['nrow'], self.domain['ncol'])).astype(variable.dtype)
        return array

    def modflow2CWATM(self, variable, correct_boundary=False):
        variable = variable.copy()
        variable[self.modflow.basin == False] = 0  
        assert not (np.isnan(variable).any())
        if correct_boundary:
            variable = variable / (self.modflow_cell_area_corrected / self.modflow_cell_area)
        cwatm_cell_area = self.var.cellArea_uncompressed.ravel()
        area = self.indices['area']
        array = (np.bincount(
            self.indices['CWatM_index'],
            weights=variable.ravel()[self.indices['ModFlow_index']] * area,
            minlength=self.var.mask.size
        ) / cwatm_cell_area).reshape(self.var.mask.shape).astype(variable.dtype)
        array[self.var.mask == 1] = np.nan
        return array

    @property
    def available_groundwater_m_modflow(self):
        groundwater_storage_available_m = (self.modflow.decompress(self.modflow.head) - (self.layer_boundaries[0] - self.max_groundwater_abstraction_depth)) * self.porosity[0]
        groundwater_storage_available_m[groundwater_storage_available_m < 0] = 0
        return groundwater_storage_available_m

    @property
    def total_groundwater_m_modflow(self):
        groundwater_storage_available_m = (self.modflow.decompress(self.modflow.head) - self.layer_boundaries[1]) * self.porosity[0]
        groundwater_storage_available_m[groundwater_storage_available_m < 0] = 0
        return groundwater_storage_available_m

    @property
    def available_groundwater_m(self):
        return self.var.compress(self.modflow2CWATM(self.available_groundwater_m_modflow))

    def initial(self):
        modflow_directory = cbinding('PathGroundwaterModflow')
        self.modflow_resolution = int(cbinding('Modflow_resolution'))

        nlay = int(loadmap('nlay'))

        with rasterio.open(cbinding('modflow_mask'), 'r') as src:
            modflow_basin = (~src.read(1).astype(np.bool))  # read in as 3-dimensional array (nlay, nrows, ncols).
            self.domain = {
                'rowsize': abs(src.profile['transform'].e),
                'colsize': abs(src.profile['transform'].a),
                'nrow': src.profile['height'],
                'ncol': src.profile['width'],
            }
        self.modflow_cell_area = self.domain['rowsize'] * self.domain['colsize']

        thickness = cbinding('thickness')
        if is_float(thickness):
            thickness = float(thickness)
            thickness = np.full((nlay, self.domain['nrow'], self.domain['ncol']), thickness)
        else:
            raise NotImplementedError

        # Coef to multiply transmissivity and storage coefficient (because ModFlow convergence is better if aquifer's thicknes is big and permeability is small)
        self.coefficient = 1

        with rasterio.open(cbinding('topo_modflow'), 'r') as src:
            topography = src.read(1).astype(np.float32)
            topography[modflow_basin == False] = np.nan

        with rasterio.open(cbinding('chanRatio'), 'r') as src:
            self.var.channel_ratio = self.var.compress(src.read(1))

        permeability_m_s = cbinding('permeability')
        if is_float(permeability_m_s):
            permeability_m_s = float(permeability_m_s)
            self.permeability = np.full((nlay, self.domain['nrow'], self.domain['ncol']), (24 * 3600 * permeability_m_s) / self.coefficient, dtype=np.float32)
        else:
            raise NotImplementedError

        self.porosity = cbinding('poro')   # default = 0.1
        if is_float(self.porosity):
            self.porosity = float(self.porosity)
            self.porosity = np.full((nlay, self.domain['nrow'], self.domain['ncol']), self.porosity, dtype=np.float32)
        else:
            raise NotImplementedError

        modflow_x = np.load(os.path.join(cbinding('cwatm_modflow_indices'), 'modflow_x.npy'))
        modflow_y = np.load(os.path.join(cbinding('cwatm_modflow_indices'), 'modflow_y.npy'))
        cwatm_x = np.load(os.path.join(cbinding('cwatm_modflow_indices'), 'cwatm_x.npy'))
        cwatm_y = np.load(os.path.join(cbinding('cwatm_modflow_indices'), 'cwatm_y.npy'))

        self.indices = {
            'area': np.load(os.path.join(cbinding('cwatm_modflow_indices'), 'area.npy')),
            'ModFlow_index': np.array(modflow_y * self.domain['ncol'] + modflow_x),
            'CWatM_index': np.array(cwatm_y * self.var.mask.shape[1] + cwatm_x)
        }

        self.modflow_cell_area = np.bincount(self.indices['ModFlow_index'], weights=self.indices['area'], minlength=self.domain['nrow'] * self.domain['ncol']).reshape(self.domain['nrow'], self.domain['ncol'])
        
        indices_cell_area = np.bincount(self.indices['CWatM_index'], weights=self.indices['area'], minlength=self.var.mask.size)
        self.indices['modflow_area'] = self.indices['area'] * (self.var.decompress(self.var.cellArea, nanvalue=0).ravel() / indices_cell_area)[self.indices['CWatM_index']]
        self.modflow_cell_area_corrected = np.bincount(self.indices['ModFlow_index'], weights=self.indices['modflow_area'], minlength=self.domain['nrow'] * self.domain['ncol']).reshape(self.domain['nrow'], self.domain['ncol'])

        indices_cell_area = np.bincount(self.indices['ModFlow_index'], weights=self.indices['modflow_area'], minlength=self.domain['nrow'] * self.domain['ncol'])
        self.indices['cwatm_area'] = self.indices['area'] * (np.bincount(self.indices['ModFlow_index'], weights=self.indices['area'], minlength=self.domain['nrow'] * self.domain['ncol']) / indices_cell_area)[self.indices['ModFlow_index']]
        self.modflow_correction = indices_cell_area.reshape(self.domain['nrow'], self.domain['ncol']) / self.modflow_cell_area_corrected

        self.cwatm_cell_area_corrected = np.bincount(self.indices['CWatM_index'], weights=self.indices['cwatm_area'], minlength=self.var.mask.size).reshape(self.var.mask.shape)
        
        indices_cell_area = np.bincount(self.indices['CWatM_index'], weights=self.indices['area'], minlength=self.var.mask.size)
        self.indices['area'] = self.indices['area'] * (self.var.cellArea_uncompressed.ravel() / indices_cell_area)[self.indices['CWatM_index']]

        soildepth_as_GWtop = returnBool('use_soildepth_as_GWtop')
        correct_depth_underlakes = returnBool('correct_soildepth_underlakes')

        if soildepth_as_GWtop:  # topographic minus soil depth map is used as groundwater upper boundary
            if correct_depth_underlakes:  # in some regions or models soil depth is around zeros under lakes, so it should be similar than neighboring cells
                print('=> Topography minus soil depth is used as upper limit of groundwater. Correcting depth under lakes.')
                waterBodyID_temp = loadmap('waterBodyID').astype(np.int64)
                soil_depth_temp = np.where(waterBodyID_temp != 0, np.nanmedian(self.var.soildepth_12) - loadmap('depth_underlakes'), self.var.soildepth_12)
                soil_depth_temp = np.where(self.var.soildepth_12 < 0.4, np.nanmedian(self.var.soildepth_12), self.var.soildepth_12)  # some cells around lake have small soil depths
                soildepth_modflow = self.CWATM2modflow(self.var.decompress(soil_depth_temp))
                soildepth_modflow[np.isnan(soildepth_modflow)] = 0
            else:
                print('=> Topography minus soil depth is used as upper limit of groundwater. No correction of depth under lakes')
                soildepth_modflow = self.CWATM2modflow(self.var.decompress(self.var.soildepth_12))
                soildepth_modflow[np.isnan(soildepth_modflow)] = 0
        else:  # topographic map is used as groundwater upper boundary
            if correct_depth_underlakes:  # we make a manual correction
                print('=> Topography is used as upper limit of groundwater. Correcting depth under lakes. It can make ModFlow difficulties to converge')
                waterBodyID_temp = loadmap('waterBodyID').astype(np.int64)
                soil_depth_temp = np.where(waterBodyID_temp != 0, loadmap('depth_underlakes'), 0)
                soildepth_modflow = self.CWATM2modflow(self.var.decompress(soil_depth_temp))
                soildepth_modflow[np.isnan(soildepth_modflow)] = 0
            else:
                print('=> Topography is used as upper limit of groundwater. No correction of depth under lakes')
                soildepth_modflow = np.zeros((self.domain['nrow'], self.domain['ncol']), dtype=np.float32)

        soildepth_modflow = self.CWATM2modflow(self.var.decompress(self.var.soildepth_12))
        soildepth_modflow[np.isnan(soildepth_modflow)] = 0

        self.var.leakageriver_factor = loadmap('leakageriver_permea')  # in m/day
        self.var.leakagelake_factor = loadmap('leakagelake_permea')  # in m/day

        self.layer_boundaries = np.empty((nlay + 1, self.domain['nrow'], self.domain['ncol']), dtype=np.float64)
        self.layer_boundaries[0] = topography - soildepth_modflow - 0.05
        self.layer_boundaries[1] = self.layer_boundaries[0] - thickness

        # defining the initial water table map (it can be a hydraulic head map previously simulated)
        self.var.load_init_water_table = returnBool('load_init_water_table')
        if self.var.load_init_water_table:
            print('=> Initial water table depth is uploaded from ', cbinding('init_water_table'))
            head = np.load(cbinding('init_water_table'))
        else:
            start_watertabledepth = loadmap('initial_water_table_depth')
            print('=> Water table depth is - ', start_watertabledepth, ' m at the begining')
            head = self.layer_boundaries[0] - start_watertabledepth  # initiate head 2 m below the top of the aquifer layer

        self.max_groundwater_abstraction_depth = cbinding('max_groundwater_abstraction_depth')

        self.modflow = ModFlowSimulation(
            self.model,
            'transient',
            modflow_directory,
            ndays=globals.dateVar['intEnd'],
            specific_storage=0,
            specific_yield=float(cbinding('poro')),
            nlay=nlay,
            nrow=self.domain['nrow'],
            ncol=self.domain['ncol'],
            rowsize=self.domain['rowsize'],
            colsize=self.domain['colsize'],
            top=self.layer_boundaries[0],
            bottom=self.layer_boundaries[1],
            basin=modflow_basin,
            head=head,
            drainage_elevation=self.layer_boundaries[0],
            permeability=self.permeability,
            complexity='SIMPLE',
            verbose=Flags['loud']            
        )

        self.corrected_cwatm_cell_area = self.get_corrected_cwatm_cell_area()
        self.corrected_modflow_cell_area = self.get_corrected_modflow_cell_area()

        self.var.capillar = self.var.full_compressed(0, dtype=np.float32)

        self.var.head = self.var.compress(self.modflow2CWATM(head))

        self.modflow_test = os.path.join(self.modflow.working_directory, 'modflow_test.csv')
        with open(self.modflow_test, 'w') as f:
            f.write('mean_head,storage_change,recharge,abstraction,outflow\n')


    def dynamic(self, groundwater_recharge, groundwater_abstraction):
        assert (groundwater_abstraction + 1e-7 >= 0).all()
        groundwater_abstraction[groundwater_abstraction < 0] = 0
        assert (groundwater_recharge >= 0).all()
        assert (groundwater_abstraction <= self.available_groundwater_m + 1e-7).all()

        groundwater_storage_pre = np.nansum(self.total_groundwater_m_modflow * self.modflow_cell_area)

        groundwater_recharge_modflow = self.CWATM2modflow(self.var.decompress(groundwater_recharge, nanvalue=0))
        assert isclose((groundwater_recharge * self.var.cellArea).sum(), np.nansum(groundwater_recharge_modflow * self.modflow_cell_area), rel_tol=1e-6)
        self.modflow.set_recharge(groundwater_recharge_modflow)
        
        groundwater_abstraction_modflow = self.CWATM2modflow(self.var.decompress(groundwater_abstraction, nanvalue=0))
        assert isclose((groundwater_abstraction * self.var.cellArea).sum(), np.nansum(groundwater_abstraction_modflow * self.modflow_cell_area), rel_tol=1e-6)
        self.modflow.set_groundwater_abstraction(groundwater_abstraction_modflow)

        self.modflow.step()

        head = self.modflow.decompress(self.modflow.head.astype(np.float32))
        self.var.head = self.var.compress(self.modflow2CWATM(head, correct_boundary=False))

        assert self.permeability.ndim == 3
        groundwater_outflow = np.where(
            head - self.layer_boundaries[0] >= 0,
            (head - self.layer_boundaries[0]) * self.coefficient * self.permeability[0],
        0)
        assert (groundwater_outflow >= 0).all()
        
        groundwater_storage_post = np.nansum(self.total_groundwater_m_modflow * self.modflow_cell_area)
        storage_change = (groundwater_storage_post - groundwater_storage_pre)
        outflow = np.nansum(groundwater_abstraction_modflow * self.modflow_cell_area) + (groundwater_outflow * self.modflow_cell_area).sum()
        inflow = np.nansum(groundwater_recharge_modflow * self.modflow_cell_area)
        with open(self.modflow_test, 'a') as f:
            f.write(f'{np.nanmean(head)},{storage_change},{np.nansum(groundwater_recharge_modflow * self.modflow_resolution ** 2)},{np.nansum(groundwater_abstraction_modflow * self.modflow_resolution ** 2)},{(groundwater_outflow * self.modflow_resolution ** 2).sum()}\n')
        if not isclose(storage_change, inflow - outflow, rel_tol=0.02) and not isclose(storage_change, inflow - outflow, abs_tol=10_000_000):
            print('modflow discrepancy', storage_change, inflow - outflow)

        groundwater_outflow_cwatm = self.var.compress(self.modflow2CWATM(groundwater_outflow, correct_boundary=True))
        assert isclose((groundwater_outflow_cwatm * self.var.cellArea).sum(), (groundwater_outflow * self.modflow_cell_area).sum(), rel_tol=0.0001)

        self.var.capillar = groundwater_outflow_cwatm * (1 - self.var.channel_ratio)
        self.var.baseflow = groundwater_outflow_cwatm * self.var.channel_ratio
        
        # capriseindex is 1 where capilary rise occurs
        self.model.subvar.capriseindex = self.model.to_subvar(data=self.var.compress(
            self.modflow2CWATM((groundwater_outflow > 0).astype(np.float32), correct_boundary=False)
        ), fn=None)
        assert (self.model.subvar.capriseindex >= 0).all() and (self.model.subvar.capriseindex <= 1).all()
