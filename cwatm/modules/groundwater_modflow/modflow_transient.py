import numpy as np
from .modflow_model import ModFlowSimulation
from pyproj import CRS, Transformer


class groundwater_modflow:
    def __init__(self, model):
        self.var = model.data.grid
        self.model = model

        # load hydraulic conductivity (md-1)
        hydraulic_conductivity = self.model.data.grid.load(
            self.model.model_structure["grid"]["groundwater/hydraulic_conductivity"],
            layer=None,
        )

        specific_yield = self.model.data.grid.load(
            self.model.model_structure["grid"]["groundwater/specific_yield"],
            layer=None,
        )

        elevation = self.model.data.grid.load(
            self.model.model_structure["grid"]["landsurface/topo/elevation"]
        )

        assert hydraulic_conductivity.shape == specific_yield.shape

        self.var.channel_ratio = self.var.load(
            self.model.model_structure["grid"]["routing/kinematic/channel_ratio"]
        )

        self.var.leakageriver_factor = 0.001  # in m/day
        self.var.leakagelake_factor = 0.001  # in m/day

        soil_depth = self.model.data.to_grid(
            HRU_data=self.model.data.HRU.soil_layer_height.sum(axis=0),
            fn="weightedmean",
        )
        bottom_soil = elevation - soil_depth

        self.initial_water_table_depth = 2
        initial_head = self.model.data.modflow.load_initial(
            "head",
            default=bottom_soil - self.initial_water_table_depth,
        )

        assert hydraulic_conductivity.shape[0] == 1, "currently only 1 layer supported"
        bottom = np.expand_dims(elevation - 100, 0)

        gt = self.model.data.grid.gt
        nrows, ncols = self.model.data.grid.mask.shape

        x_coordinates = np.arange(gt[0], gt[0] + gt[1] * (ncols + 1), gt[1])
        y_coordinates = np.arange(gt[3], gt[3] + gt[5] * (nrows + 1), gt[5])

        center_longitude = (x_coordinates[0] + x_coordinates[-1]) / 2
        center_latitude = (y_coordinates[0] + y_coordinates[-1]) / 2

        utm_crs = CRS.from_dict(
            {
                "proj": "utm",
                "ellps": "WGS84",
                "lat_0": center_latitude,
                "lon_0": center_longitude,
                "zone": int((center_longitude + 180) / 6) + 1,
            }
        )

        # Create a topography 2D map
        x_vertices, y_vertices = np.meshgrid(x_coordinates, y_coordinates)

        # convert to modflow coordinates
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

        # Transform the points
        x_transformed, y_transformed = transformer.transform(
            x_vertices.ravel(), y_vertices.ravel()
        )

        # Reshape back to the original grid shape
        x_transformed = x_transformed.reshape(x_vertices.shape)
        y_transformed = y_transformed.reshape(y_vertices.shape)

        self.modflow = ModFlowSimulation(
            self.model,
            "transient",
            ndays=self.model.n_timesteps,
            x_coordinates_vertices=x_transformed,
            y_coordinates_vertices=y_transformed,
            specific_storage=np.zeros_like(specific_yield),
            specific_yield=specific_yield,
            topography=elevation,
            bottom_soil=bottom_soil,
            bottom=bottom,
            basin_mask=self.model.data.grid.mask,
            head=initial_head,
            hydraulic_conductivity=hydraulic_conductivity,
            complexity="SIMPLE",
            verbose=False,
        )

        self.var.capillar = self.var.load_initial(
            "capillar", default=self.var.full_compressed(0, dtype=np.float32)
        )

    def step(self, groundwater_recharge, groundwater_abstraction):
        assert (groundwater_abstraction + 1e-7 >= 0).all()
        groundwater_abstraction[groundwater_abstraction < 0] = 0
        assert (groundwater_recharge >= 0).all()

        groundwater_storage_pre = self.modflow.groundwater_content_m3.sum()

        self.modflow.set_recharge_m(groundwater_recharge)
        self.modflow.set_groundwater_abstraction_m(groundwater_abstraction)
        self.modflow.step()

        drainage_m3 = self.modflow.drainage_m3.sum()
        groundwater_abstraction_m3 = groundwater_abstraction * self.var.cellArea
        recharge_m3 = groundwater_recharge * self.var.cellArea
        groundwater_storage_post = self.modflow.groundwater_content_m3.sum()

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

        groundwater_drainage = self.modflow.drainage_m3 / self.var.cellArea

        self.var.capillar = groundwater_drainage * (1 - self.var.channel_ratio)
        self.var.baseflow = groundwater_drainage * self.var.channel_ratio

        # capriseindex is 1 where capilary rise occurs
        self.model.data.HRU.capriseindex = self.model.data.to_HRU(
            data=groundwater_drainage > 0
        )
