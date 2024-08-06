from time import time
from contextlib import contextmanager
import os
import numpy as np
from xmipy import XmiWrapper
import flopy
import json
import hashlib
import platform


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


class ModFlowSimulation:

    def __init__(
        self,
        model,
        name,
        ndays,
        specific_storage,
        specific_yield,
        nlay,
        nrow,
        ncol,
        row_resolution,
        col_resolution,
        topography,
        bottom_soil,
        bottom,
        basin_mask,
        head,
        hydraulic_conductivity,
        complexity="COMPLEX",
        verbose=False,
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.nrow = nrow
        self.ncol = ncol
        self.row_resolution = row_resolution
        self.col_resolution = col_resolution
        self.basin_mask = basin_mask
        assert self.basin_mask.dtype == bool
        self.n_active_cells = self.basin_mask.size - self.basin_mask.sum()
        self.working_directory = model.simulation_root / "modflow_model"
        os.makedirs(self.working_directory, exist_ok=True)
        self.verbose = verbose

        self.topography = topography
        self.bottom = bottom
        assert (
            self.bottom.ndim == 3
        ), "Bottom should be a 3D array of shape (nlay, nrow, ncol), representing the bottom of the aquifer"
        self.specific_yield = specific_yield
        assert self.specific_yield.shape == self.bottom.shape

        arguments = dict(locals())
        arguments.pop("self")
        arguments.pop("model")
        self.hash_file = os.path.join(self.working_directory, "input_hash")

        save_flows = False

        if not self.load_from_disk(arguments):
            try:
                if self.verbose:
                    print("Creating MODFLOW model")
                sim = flopy.mf6.MFSimulation(
                    sim_name=self.name,
                    version="mf6",
                    exe_name=os.path.join("modflow", "mf6"),
                    sim_ws=os.path.realpath(self.working_directory),
                    memory_print_option="all",
                )

                # create tdis package
                tdis = flopy.mf6.ModflowTdis(
                    sim, nper=ndays, perioddata=[(1.0, 1, 1)] * ndays
                )

                # create iterative model solution and register the gwf model with it
                ims = flopy.mf6.ModflowIms(
                    sim,
                    print_option=None,
                    complexity=complexity,
                    linear_acceleration="BICGSTAB",
                )

                # create gwf model
                gwf = flopy.mf6.ModflowGwf(
                    sim,
                    modelname=self.name,
                    newtonoptions="under_relaxation",
                    print_input=save_flows,
                    print_flows=save_flows,
                )

                discretization = flopy.mf6.ModflowGwfdis(
                    gwf,
                    nlay=nlay,
                    nrow=self.nrow,
                    ncol=self.ncol,
                    delr=self.row_resolution,
                    delc=self.col_resolution,
                    top=self.topography,
                    botm=self.bottom,
                    idomain=~self.basin_mask,
                    nogrb=True,
                )

                initial_conditions = flopy.mf6.ModflowGwfic(gwf, strt=head)
                node_property_flow = flopy.mf6.ModflowGwfnpf(
                    gwf, save_flows=save_flows, icelltype=1, k=hydraulic_conductivity
                )

                storage = flopy.mf6.ModflowGwfsto(
                    gwf,
                    save_flows=save_flows,
                    iconvert=1,
                    ss=specific_storage,  # specific storage
                    sy=specific_yield,  # specific yield
                    steady_state=False,
                    transient=True,
                )

                recharge = np.zeros((self.n_active_cells, 4), dtype=np.float32)
                recharge_locations = np.where(
                    self.basin_mask == False
                )  # only set wells where basin mask is False
                # 0: layer, 1: y-idx, 2: x-idx, 3: rate
                recharge[:, 0] = 0
                recharge[:, 1] = recharge_locations[0]
                recharge[:, 2] = recharge_locations[1]
                recharge[:, 3] = 0.0
                recharge = recharge.tolist()
                recharge = [[(int(i), int(j), int(k)), l] for i, j, k, l in recharge]

                recharge = flopy.mf6.ModflowGwfrch(
                    gwf,
                    fixed_cell=False,
                    print_input=False,
                    print_flows=False,
                    save_flows=save_flows,
                    boundnames=None,
                    maxbound=self.n_active_cells,
                    stress_period_data=recharge,
                )

                wells = np.zeros((self.n_active_cells, 4), dtype=np.int32)
                well_locations = np.where(
                    self.basin_mask == False
                )  # only set wells where basin is False
                # 0: layer, 1: y-idx, 2: x-idx, 3: rate
                wells[:, 1] = well_locations[0]
                wells[:, 2] = well_locations[1]
                wells = wells.tolist()

                wells = flopy.mf6.ModflowGwfwel(
                    gwf,
                    print_input=False,
                    print_flows=False,
                    save_flows=save_flows,
                    maxbound=self.n_active_cells,
                    stress_period_data=wells,
                    boundnames=False,
                    auto_flow_reduce=0.1,
                )

                drainage = np.zeros(
                    (self.n_active_cells, 5)
                )  # Only i,j,k indices should be integer
                drainage_locations = np.where(
                    self.basin_mask == False
                )  # only set wells where basin is True
                # 0: layer, 1: y-idx, 2: x-idx, 3: drainage altitude, 4: permeability
                drainage[:, 0] = 0
                drainage[:, 1] = drainage_locations[0]
                drainage[:, 2] = drainage_locations[1]
                drainage[:, 3] = bottom_soil[
                    drainage_locations
                ]  # This one should not be an integer
                # the 0-th layer is the top layer, so the conductivity is the same as the top layer
                drainage[:, 4] = (
                    hydraulic_conductivity[0, self.basin_mask == False]
                    * self.row_resolution
                    * self.col_resolution
                )
                drainage = drainage.tolist()
                drainage = [
                    [(int(layer), int(row), int(column)), elevation, conductivity]
                    for layer, row, column, elevation, conductivity in drainage
                ]

                drainage = flopy.mf6.ModflowGwfdrn(
                    gwf,
                    maxbound=self.n_active_cells,
                    stress_period_data=drainage,
                    print_input=save_flows,
                    print_flows=save_flows,
                    save_flows=save_flows,
                )

                sim.write_simulation()
                self.write_hash_to_disk()
            except:
                if os.path.exists(self.hash_file):
                    os.remove(self.hash_file)
                raise
            # sim.run_simulation()
        elif self.verbose:
            print("Loading MODFLOW model from disk")

        self.load_bmi()

    def write_hash_to_disk(self):
        with open(self.hash_file, "wb") as f:
            f.write(self.hash)

    def load_from_disk(self, arguments):
        hashable_dict = {}
        for key, value in arguments.items():
            if isinstance(value, np.ndarray):
                value = str(value.tobytes())
            hashable_dict[key] = value

        self.hash = hashlib.md5(
            json.dumps(hashable_dict, sort_keys=True).encode()
        ).digest()
        if not os.path.exists(self.hash_file):
            prev_hash = None
        else:
            with open(self.hash_file, "rb") as f:
                prev_hash = f.read().strip()
        if prev_hash == self.hash:
            if os.path.exists(os.path.join(self.working_directory, "mfsim.nam")):
                return True
            else:
                return False
        else:
            return False

    def bmi_return(self, success, model_ws):
        """
        parse libmf6.so and libmf6.dll stdout file
        """
        fpth = os.path.join("mfsim.stdout")
        with open(fpth) as f:
            lines = f.readlines()
        return success, lines

    def load_bmi(self):
        """Load the Basic Model Interface"""
        success = False

        # Current model version 6.4.2 from https://github.com/MODFLOW-USGS/modflow6/releases/tag/6.4.2
        if platform.system() == "Windows":
            libary_name = "windows/libmf6.dll"
        elif platform.system() == "Linux":
            libary_name = "linux/libmf6.so"
        elif platform.system() == "Darwin":
            libary_name = "mac/libmf6.dylib"
        else:
            raise ValueError(f"Platform {platform.system()} not recognized.")

        with cd(self.working_directory):
            # modflow requires the real path (no symlinks etc.)
            library_path = os.path.realpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), libary_name)
            )
            assert os.path.exists(library_path)
            try:
                self.mf6 = XmiWrapper(library_path)
            except Exception as e:
                print("Failed to load " + library_path)
                print("with message: " + str(e))
                self.bmi_return(success, self.working_directory)
                raise

            # modflow requires the real path (no symlinks etc.)
            config_file = os.path.realpath("mfsim.nam")
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    f"Config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?"
                )

            # initialize the model
            try:
                self.mf6.initialize(config_file)
            except:
                self.bmi_return(success, self.working_directory)
                raise

            if self.verbose:
                print("MODFLOW model initialized")

        self.end_time = self.mf6.get_end_time()

        self.prepare_time_step()

    @property
    def head(self):
        head_tag = self.mf6.get_var_address("X", self.name)
        return self.mf6.get_value_ptr(head_tag)

    @property
    def groundwater_depth(self):
        return self.topography - self.decompress(self.head)

    @property
    def groundwater_content_m(self):
        # use the bottom of the bottom layer
        return (self.decompress(self.head) - self.bottom[-1]) * self.specific_yield[0]

    @property
    def groundwater_content_m3(self):
        return self.groundwater_content_m * self.row_resolution * self.col_resolution

    @property
    def well_rate(self):
        well_tag = self.mf6.get_var_address("BOUND", self.name, "WEL_0")
        return self.mf6.get_value_ptr(well_tag)[:, 0]

    @well_rate.setter
    def well_rate(self, value):
        well_tag = self.mf6.get_var_address("BOUND", self.name, "WEL_0")
        self.mf6.get_value_ptr(well_tag)[:, 0][:] = value

    @property
    def drainage_m3(self):
        drainage_tag = self.mf6.get_var_address("SIMVALS", self.name, "DRN_0")
        drainage = self.mf6.get_value_ptr(drainage_tag)
        drainage = self.decompress(drainage)
        return -drainage

    @property
    def drainage_m(self):
        return self.drainage_m3 / (self.row_resolution * self.col_resolution)

    @property
    def recharge_m3(self):
        recharge_tag = self.mf6.get_var_address("BOUND", self.name, "RCH_0")
        recharge = self.mf6.get_value_ptr(recharge_tag)[:, 0]
        recharge = self.decompress(recharge)
        return recharge

    @property
    def recharge_m(self):
        return self.recharge_m3 / (self.row_resolution * self.col_resolution)

    @recharge_m3.setter
    def recharge_m3(self, value):
        recharge_tag = self.mf6.get_var_address("BOUND", self.name, "RCH_0")
        self.mf6.get_value_ptr(recharge_tag)[:, 0][:] = value

    @property
    def max_iter(self):
        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        return self.mf6.get_value_ptr(mxit_tag)[0]

    def compress(self, a):
        return a[~self.basin_mask]

    def decompress(self, a):
        o = np.full(self.basin_mask.shape, np.nan, dtype=a.dtype)
        o[~self.basin_mask] = a
        return o

    def prepare_time_step(self):
        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    def set_recharge_m(self, recharge):
        """Set recharge, value in m/day"""
        self.recharge_m3 = recharge[~self.basin_mask] * (
            self.row_resolution * self.col_resolution
        )

    def set_groundwater_abstraction_m(self, groundwater_abstraction):
        """Set well rate, value in m/day"""
        well_rate = -groundwater_abstraction[
            ~self.basin_mask
        ]  # modflow well rate is negative if abstraction occurs
        assert (well_rate <= 0).all()
        self.well_rate = well_rate * (self.row_resolution * self.col_resolution)

    def step(self, plot=False):
        if self.mf6.get_current_time() > self.end_time:
            raise StopIteration(
                "MODFLOW used all iteration steps. Consider increasing `ndays`"
            )

        t0 = time()
        # loop over subcomponents
        n_solutions = self.mf6.get_subcomponent_count()
        for solution_id in range(1, n_solutions + 1):

            # convergence loop
            kiter = 0
            self.mf6.prepare_solve(solution_id)
            while kiter < self.max_iter:
                has_converged = self.mf6.solve(solution_id)
                kiter += 1

                if has_converged:
                    break

            self.mf6.finalize_solve(solution_id)

        self.mf6.finalize_time_step()

        if self.verbose:
            print(
                f"MODFLOW timestep {int(self.mf6.get_current_time())} converged in {round(time() - t0, 2)} seconds"
            )

        # If next step exists, prepare timestep. Otherwise the data set through the bmi
        # will be overwritten when preparing the next timestep.
        if self.mf6.get_current_time() < self.end_time:
            self.prepare_time_step()

    def finalize(self):
        self.mf6.finalize()
