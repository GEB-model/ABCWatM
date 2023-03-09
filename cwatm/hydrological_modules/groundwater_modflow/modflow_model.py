from time import time
from contextlib import contextmanager
import os
import numpy as np
from xmipy import XmiWrapper
import flopy
import json
import hashlib
import platform
from cwatm.management_modules.globals import outDir

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
        folder,
        ndays,
        specific_storage,
        specific_yield,
        nlay,
        nrow,
        ncol,
        rowsize,
        colsize,
        top,
        bottom,
        basin,
        head,
        drainage_elevation,
        permeability,
        complexity='COMPLEX',
        verbose=False
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.folder = folder
        self.nrow = nrow
        self.ncol = ncol
        self.rowsize = rowsize
        self.colsize = colsize
        self.basin = basin
        self.n_active_cells = self.basin.sum()
        self.working_directory = os.path.join(outDir['OUTPUT'], 'modflow_model')
        os.makedirs(self.working_directory, exist_ok=True)
        self.verbose = verbose

        arguments = dict(locals())
        arguments.pop('self')
        arguments.pop('model')
        self.hash_file = os.path.join(self.working_directory, 'input_hash')

        if not self.load_from_disk(arguments):
            try:
                if self.verbose:
                    print("Creating MODFLOW model")
                sim = flopy.mf6.MFSimulation(
                    sim_name=self.name,
                    version='mf6',
                    exe_name=os.path.join(folder, 'mf6'),
                    sim_ws=os.path.realpath(self.working_directory),
                    memory_print_option='all'
                )
                
                # create tdis package
                tdis = flopy.mf6.ModflowTdis(sim, nper=ndays, perioddata=[(1.0, 1, 1)] * ndays)

                # create iterative model solution and register the gwf model with it
                ims = flopy.mf6.ModflowIms(sim, print_option=None, complexity=complexity, linear_acceleration='BICGSTAB')

                # create gwf model
                gwf = flopy.mf6.ModflowGwf(sim, modelname=self.name, newtonoptions='under_relaxation', print_input=False, print_flows=False)

                discretization = flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=self.nrow, ncol=self.ncol,
                    delr=self.rowsize, delc=self.colsize, top=top,
                    botm=bottom, idomain=self.basin, nogrb=True)

                initial_conditions = flopy.mf6.ModflowGwfic(gwf, strt=head)
                node_property_flow = flopy.mf6.ModflowGwfnpf(gwf, save_flows=True, icelltype=1, k=permeability)

                output_control = flopy.mf6.ModflowGwfoc(
                    gwf,
                    head_filerecord=f'{self.name}.hds',saverecord=[('HEAD', 'FREQUENCY', 10)])

                storage = flopy.mf6.ModflowGwfsto(gwf,
                    save_flows=False,
                    iconvert=1,
                    ss=specific_storage,  # specific storage
                    sy=specific_yield,  # specific yield
                    steady_state=False,
                    transient=True,
                )

                recharge = np.zeros((self.basin.sum(), 4), dtype=np.int32)
                recharge_locations = np.where(self.basin == True)  # only set wells where basin is True
                # 0: layer, 1: y-idx, 2: x-idx, 3: rate
                recharge[:, 0] = 0
                recharge[:, 1] = recharge_locations[0]
                recharge[:, 2] = recharge_locations[1]
                recharge[:, 3] = 0
                recharge = recharge.tolist()
                recharge = [[(int(i), int(j), int(k)), l] for i, j, k, l in recharge]

                recharge = flopy.mf6.ModflowGwfrch(gwf, fixed_cell=False,
                                print_input=False, print_flows=False,
                                save_flows=False, boundnames=None,
                                maxbound=self.basin.sum(), stress_period_data=recharge)

                wells = np.zeros((self.basin.sum(), 4), dtype=np.int32)
                well_locations = np.where(self.basin == True)  # only set wells where basin is True
                # 0: layer, 1: y-idx, 2: x-idx, 3: rate
                wells[:, 1] = well_locations[0]
                wells[:, 2] = well_locations[1]
                wells = wells.tolist()

                wells = flopy.mf6.ModflowGwfwel(gwf, print_input=False, print_flows=False, save_flows=False,
                                            maxbound=self.basin.sum(), stress_period_data=wells,
                                            boundnames=False, auto_flow_reduce=0.1)

                drainage = np.zeros((self.basin.sum(), 5))  # Only i,j,k indices should be integer
                #drainage = np.zeros((self.basin.sum(), 5), dtype=np.int32)
                drainage_locations = np.where(self.basin == True)  # only set wells where basin is True
                # 0: layer, 1: y-idx, 2: x-idx, 3: drainage altitude, 4: permeability
                drainage[:, 0] = 0
                drainage[:, 1] = drainage_locations[0]
                drainage[:, 2] = drainage_locations[1]
                drainage[:, 3] = drainage_elevation[drainage_locations] # This one should not be an integer
                drainage[:, 4] = permeability[0, self.basin == True] * self.rowsize * self.colsize
                drainage = drainage.tolist()
                drainage = [[(int(i), int(j), int(k)), l, m] for i, j, k, l, m in drainage]

                drainage = flopy.mf6.ModflowGwfdrn(gwf, maxbound=self.basin.sum(), stress_period_data=drainage,
                                            print_input=False, print_flows=False, save_flows=False)
                
                sim.write_simulation()
            except:
                if os.path.exists(self.hash_file):
                    os.remove(self.hash_file)
                raise
            # sim.run_simulation()
        elif self.verbose:
            print("Loading MODFLOW model from disk")
        
        self.load_bmi()

    def load_from_disk(self, arguments):
        hashable_dict = {}
        for key, value in arguments.items():
            if isinstance(value, np.ndarray):
                value = str(value.tobytes())
            hashable_dict[key] = value

        current_hash = hashlib.md5(json.dumps(hashable_dict, sort_keys=True).encode()).digest()
        if not os.path.exists(self.hash_file):
            prev_hash = None
        else:
            with open(self.hash_file, 'rb') as f:
                prev_hash = f.read().strip()
        if prev_hash == current_hash:
            if os.path.exists(os.path.join(self.working_directory, 'mfsim.nam')):
                return True
            else:
                return False
        else:
            with open(self.hash_file, 'wb') as f:
                f.write(current_hash)
            return False

    def bmi_return(self, success, model_ws):
        """
        parse libmf6.so and libmf6.dll stdout file
        """
        fpth = os.path.join('mfsim.stdout')
        return success, open(fpth).readlines()

    def load_bmi(self):
        """Load the Basic Model Interface"""
        success = False
                
        if platform.system() == 'Windows':
            libary_name = 'libmf6.dll'
        elif platform.system() == 'Linux':
            libary_name = 'libmf6.so'
        else:
            raise ValueError(f'Platform {platform.system()} not recognized.')

        with cd(self.working_directory):
            # modflow requires the real path (no symlinks etc.)
            library_path = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), libary_name))
            assert os.path.exists(library_path)
            try:
                self.mf6 = XmiWrapper(library_path)
            except Exception as e:
                print("Failed to load " + library_path)
                print("with message: " + str(e))
                self.bmi_return(success, self.working_directory)
                raise

            # modflow requires the real path (no symlinks etc.)
            config_file = os.path.realpath('mfsim.nam')
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?")

            # initialize the model
            try:
                self.mf6.initialize(config_file)
            except:
                self.bmi_return(success, self.working_directory)
                raise

            if self.verbose:
                print("MODFLOW model initialized")
        
        self.end_time = self.mf6.get_end_time()

        recharge_tag = self.mf6.get_var_address("BOUND", self.name, "RCH_0")
        # there seems to be a bug in xmipy where the size of the pointer to RCHA is
        # is the size of the entire modflow area, including basined cells. Only the first
        # part of the array is actually used, when a part of the area is basined. Since
        # numpy returns a view of the array when the array[]-syntax is used, we can simply
        # use the view of the first part of the array up to the number of active
        # (non-basined) cells
        self.recharge = self.mf6.get_value_ptr(recharge_tag)[:, 0]
        
        head_tag = self.mf6.get_var_address("X", self.name)
        self.head = self.mf6.get_value_ptr(head_tag)

        well_tag = self.mf6.get_var_address("BOUND", self.name, "WEL_0")
        self.well_rate = self.mf6.get_value_ptr(well_tag)[:, 0]

        drainage_tag = self.mf6.get_var_address("BOUND", self.name, "DRN_0")
        self.drainage = self.mf6.get_value_ptr(drainage_tag)[:, 1]

        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        self.max_iter = self.mf6.get_value_ptr(mxit_tag)[0]

        self.prepare_time_step()

    def compress(self, a):
        return a[self.basin]

    def decompress(self, a):
        o = np.full(self.basin.shape, np.nan, dtype=a.dtype)
        o[self.basin] = a
        return o

    def prepare_time_step(self):
        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    def set_recharge(self, recharge):
        """Set recharge, value in m/day"""
        recharge = recharge[self.basin == True]
        self.recharge[:] = recharge * (self.rowsize * self.colsize)
    
    def set_groundwater_abstraction(self, groundwater_abstraction):
        """Set well rate, value in m/day"""
        well_rate = - groundwater_abstraction[self.basin == True]  # modflow well rate is negative if abstraction occurs
        assert (well_rate <= 0).all()
        self.well_rate[:] = well_rate * (self.rowsize * self.colsize)

    def get_drainage(self):
        """Get Modflow drainage in m/day"""
        return self.decompress(self.drainage / (self.rowsize * self.colsize))

    def step(self, plot=False):
        if self.mf6.get_current_time() > self.end_time:
            raise StopIteration("MODFLOW used all iteration steps. Consider increasing `ndays`")

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
            print(f'MODFLOW timestep {int(self.mf6.get_current_time())} converged in {round(time() - t0, 2)} seconds')
        
        # If next step exists, prepare timestep. Otherwise the data set through the bmi
        # will be overwritten when preparing the next timestep.
        if self.mf6.get_current_time() < self.end_time:
            self.prepare_time_step()

    def finalize(self):
        self.mf6.finalize()