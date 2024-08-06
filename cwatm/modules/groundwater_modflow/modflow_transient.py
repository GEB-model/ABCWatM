from math import isclose
import numpy as np
from .modflow_model import (
    ModFlowSimulation,
)
import rasterio


class groundwater_modflow:
    def __init__(self, model):
        self.var = model.data.grid
        self.model = model

        self.depth_underlakes = 1.5

        with rasterio.open(
            self.model.model_structure["MODFLOW_grid"][
                "groundwater/modflow/modflow_mask"
            ],
            "r",
        ) as src:
            self.modflow_basin_mask = ~src.read(1).astype(bool)
            self.domain = {
                "row_resolution": abs(src.profile["transform"].e),
                "col_resolution": abs(src.profile["transform"].a),
                "nrow": src.profile["height"],
                "ncol": src.profile["width"],
            }

        # load hydraulic conductivity (md-1)
        with rasterio.open(
            self.model.model_structure["MODFLOW_grid"][
                "groundwater/modflow/hydraulic_conductivity"
            ],
            "r",
        ) as src:
            self.hydraulic_conductivity = src.read().astype(np.float32)
            self.hydraulic_conductivity[:, self.modflow_basin_mask == True] = np.nan

        assert self.hydraulic_conductivity.ndim == 3
        nlay = self.hydraulic_conductivity.shape[0]

        # load specific yield
        with rasterio.open(
            self.model.model_structure["MODFLOW_grid"][
                "groundwater/modflow/specific_yield"
            ],
            "r",
        ) as src:
            self.specific_yield = src.read().astype(np.float32)
            self.specific_yield[:, self.modflow_basin_mask == True] = np.nan

        assert self.hydraulic_conductivity.shape == self.specific_yield.shape

        thickness = 100.0
        thickness = np.full((nlay, self.domain["nrow"], self.domain["ncol"]), thickness)

        # load elevation
        with rasterio.open(
            self.model.model_structure["MODFLOW_grid"][
                "groundwater/modflow/modflow_elevation"
            ],
            "r",
        ) as src:
            topography = src.read(1).astype(np.float32)
            topography[self.modflow_basin_mask == True] = np.nan

        # load chanRatio
        self.var.channel_ratio = self.var.load(
            self.model.model_structure["grid"]["routing/kinematic/channel_ratio"]
        )

        assert self.hydraulic_conductivity.ndim == 3
        assert self.hydraulic_conductivity.shape[0] == nlay

        modflow_x = np.load(
            self.model.model_structure["binary"]["groundwater/modflow/x_modflow"]
        )["data"]
        modflow_y = np.load(
            self.model.model_structure["binary"]["groundwater/modflow/y_modflow"]
        )["data"]
        x_hydro = np.load(
            self.model.model_structure["binary"]["groundwater/modflow/x_hydro"]
        )["data"]
        y_hydro = np.load(
            self.model.model_structure["binary"]["groundwater/modflow/y_hydro"]
        )["data"]

        self.indices = {
            "area": np.load(
                self.model.model_structure["binary"]["groundwater/modflow/area"]
            )["data"],
            "ModFlow_index": np.array(modflow_y * self.domain["ncol"] + modflow_x),
            "CWatM_index": np.array(y_hydro * self.var.mask.shape[1] + x_hydro),
        }

        self.modflow_cell_area = np.bincount(
            self.indices["ModFlow_index"],
            weights=self.indices["area"],
            minlength=self.domain["nrow"] * self.domain["ncol"],
        ).reshape(self.domain["nrow"], self.domain["ncol"])

        indices_cell_area = np.bincount(
            self.indices["CWatM_index"],
            weights=self.indices["area"],
            minlength=self.var.mask.size,
        )
        self.indices["modflow_area"] = (
            self.indices["area"]
            * (
                self.var.decompress(self.var.cellArea, fillvalue=0).ravel()
                / indices_cell_area
            )[self.indices["CWatM_index"]]
        )
        self.modflow_cell_area_corrected = np.bincount(
            self.indices["ModFlow_index"],
            weights=self.indices["modflow_area"],
            minlength=self.domain["nrow"] * self.domain["ncol"],
        ).reshape(self.domain["nrow"], self.domain["ncol"])

        indices_cell_area = np.bincount(
            self.indices["ModFlow_index"],
            weights=self.indices["modflow_area"],
            minlength=self.domain["nrow"] * self.domain["ncol"],
        )
        self.indices["cwatm_area"] = (
            self.indices["area"]
            * (
                np.bincount(
                    self.indices["ModFlow_index"],
                    weights=self.indices["area"],
                    minlength=self.domain["nrow"] * self.domain["ncol"],
                )
                / indices_cell_area
            )[self.indices["ModFlow_index"]]
        )
        self.modflow_correction = (
            indices_cell_area.reshape(self.domain["nrow"], self.domain["ncol"])
            / self.modflow_cell_area_corrected
        )

        self.cwatm_cell_area_corrected = np.bincount(
            self.indices["CWatM_index"],
            weights=self.indices["cwatm_area"],
            minlength=self.var.mask.size,
        ).reshape(self.var.mask.shape)

        indices_cell_area = np.bincount(
            self.indices["CWatM_index"],
            weights=self.indices["area"],
            minlength=self.var.mask.size,
        )
        self.indices["area"] = (
            self.indices["area"]
            * (self.var.cell_area_uncompressed.ravel() / indices_cell_area)[
                self.indices["CWatM_index"]
            ]
        )

        self.model.data.grid.total_soil_depth = self.model.data.to_grid(
            HRU_data=self.model.data.HRU.soil_layer_height.sum(axis=0),
            fn="weightedmean",
        )

        soildepth_modflow = self.CWATM2modflow(
            self.var.decompress(self.var.total_soil_depth)
        )

        self.var.leakageriver_factor = 0.001  # in m/day
        self.var.leakagelake_factor = 0.001  # in m/day

        self.initial_water_table_depth = 2
        head = self.model.data.modflow.load_initial(
            "head",
            default=topography - soildepth_modflow - self.initial_water_table_depth,
        )

        self.modflow = ModFlowSimulation(
            self.model,
            "transient",
            ndays=self.model.n_timesteps,
            specific_storage=0,
            specific_yield=self.specific_yield,
            nlay=nlay,
            nrow=self.domain["nrow"],
            ncol=self.domain["ncol"],
            row_resolution=self.domain["row_resolution"],
            col_resolution=self.domain["col_resolution"],
            topography=topography,
            bottom_soil=topography - soildepth_modflow,
            bottom=topography - thickness,
            basin_mask=self.modflow_basin_mask,
            head=head,
            hydraulic_conductivity=self.hydraulic_conductivity,
            complexity="SIMPLE",
            verbose=False,
        )

        self.corrected_cwatm_cell_area = self.get_corrected_cwatm_cell_area()
        self.corrected_modflow_cell_area = self.get_corrected_modflow_cell_area()

        self.var.capillar = self.var.full_compressed(0, dtype=np.float32)

        self.var.head = self.var.compress(self.modflow2CWATM(head))
        self.var.groundwater_depth = self.var.compress(
            self.modflow2CWATM(head - topography)
        )

    def get_corrected_modflow_cell_area(self):
        return np.bincount(
            self.indices["ModFlow_index"],
            weights=np.invert(self.var.mask.astype(bool)).ravel()[
                self.indices["CWatM_index"]
            ]
            * self.indices["area"],
            minlength=self.modflow.basin_mask.size,
        ).reshape(self.modflow.basin_mask.shape)

    def get_corrected_cwatm_cell_area(self):
        return (
            self.var.cell_area_uncompressed.ravel()
            - np.bincount(
                self.indices["CWatM_index"],
                weights=self.modflow.basin_mask.ravel()[self.indices["ModFlow_index"]]
                * self.indices["area"],
                minlength=self.var.mask.size,
            )
        ).reshape(self.var.mask.shape)

    def CWATM2modflow(self, variable, correct_boundary=False):
        if correct_boundary:
            modflow_cell_area = self.modflow_cell_area_corrected
            area = self.indices["modflow_area"]
        else:
            modflow_cell_area = self.modflow_cell_area
            area = self.indices["area"]
        variable[self.var.mask == 1] = 0
        assert not (np.isnan(variable).any())
        array = (
            (
                np.bincount(
                    self.indices["ModFlow_index"],
                    variable.ravel()[self.indices["CWatM_index"]] * area,
                    minlength=self.domain["nrow"] * self.domain["ncol"],
                )
            )
            .reshape((self.domain["nrow"], self.domain["ncol"]))
            .astype(variable.dtype)
        )

        # just make sure that there
        assert (array[(modflow_cell_area == 0)] == 0).all()
        array = array / modflow_cell_area
        array[self.modflow_basin_mask] = 0
        return array

    def modflow2CWATM(self, variable, correct_boundary=False):
        variable = variable.copy()
        variable[self.modflow.basin_mask == True] = 0
        assert not (np.isnan(variable).any())
        if correct_boundary:
            variable = variable / (
                self.modflow_cell_area_corrected / self.modflow_cell_area
            )
        cwatm_cell_area = self.var.cell_area_uncompressed.ravel()
        area = self.indices["area"]
        array = (
            (
                np.bincount(
                    self.indices["CWatM_index"],
                    weights=variable.ravel()[self.indices["ModFlow_index"]] * area,
                    minlength=self.var.mask.size,
                )
                / cwatm_cell_area
            )
            .reshape(self.var.mask.shape)
            .astype(variable.dtype)
        )
        array[self.var.mask == 1] = np.nan
        return array

    @property
    def available_groundwater_m(self):
        return self.var.compress(self.modflow2CWATM(self.modflow.groundwater_content_m))

    def step(self, groundwater_recharge, groundwater_abstraction):
        assert (groundwater_abstraction + 1e-7 >= 0).all()
        groundwater_abstraction[groundwater_abstraction < 0] = 0
        assert (groundwater_recharge >= 0).all()

        groundwater_storage_pre = np.nansum(self.modflow.groundwater_content_m3)

        groundwater_recharge_modflow = self.CWATM2modflow(
            self.var.decompress(groundwater_recharge, fillvalue=0)
        )
        self.modflow.set_recharge_m(groundwater_recharge_modflow)

        groundwater_abstraction_modflow = self.CWATM2modflow(
            self.var.decompress(groundwater_abstraction, fillvalue=0)
        )
        self.modflow.set_groundwater_abstraction_m(groundwater_abstraction_modflow)

        self.modflow.step()

        self.model.data.modflow.head = self.modflow.decompress(
            self.modflow.head.astype(np.float32)
        )
        self.var.head = self.var.compress(
            self.modflow2CWATM(self.model.data.modflow.head, correct_boundary=False)
        )

        self.var.groundwater_depth = self.var.compress(
            self.modflow2CWATM(self.modflow.groundwater_depth)
        )

        drainage_m3 = np.nansum(self.modflow.drainage_m3)
        groundwater_abstraction_m3 = np.nansum(
            groundwater_abstraction_modflow
            * (self.domain["col_resolution"] * self.domain["row_resolution"])
        )
        recharge_m3 = np.nansum(
            groundwater_recharge_modflow
            * (self.domain["col_resolution"] * self.domain["row_resolution"])
        )
        groundwater_storage_post = np.nansum(self.modflow.groundwater_content_m3)

        self.model.waterbalance_module.waterBalanceCheck(
            name="groundwater",
            how="sum",
            influxes=[recharge_m3],
            outfluxes=[
                groundwater_abstraction_m3,
                drainage_m3,
            ],
            prestorages=[groundwater_storage_pre],
            poststorages=[groundwater_storage_post],
            tollerance=1e-6,
        )

        groundwater_drainage = self.modflow.drainage_m3 / self.modflow_cell_area

        groundwater_drainage_cwatm = self.var.compress(
            self.modflow2CWATM(groundwater_drainage, correct_boundary=True)
        )

        self.var.capillar = groundwater_drainage_cwatm * (1 - self.var.channel_ratio)
        self.var.baseflow = groundwater_drainage_cwatm * self.var.channel_ratio

        # capriseindex is 1 where capilary rise occurs
        self.model.data.HRU.capriseindex = self.model.data.to_HRU(
            data=self.var.compress(
                self.modflow2CWATM(
                    (groundwater_drainage > 0).astype(np.float32),
                    correct_boundary=False,
                )
            ),
            fn=None,
        )
        assert (self.model.data.HRU.capriseindex >= 0).all() and (
            self.model.data.HRU.capriseindex <= 1
        ).all()
