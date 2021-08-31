# -------------------------------------------------------------------------
# Name:        INITCONDITION
# Purpose:	   Read/write initial condtions for warm start
#
# Author:      PB
#
# Created:     19/08/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import os
import numpy as np
from operator import attrgetter
from cwatm.management_modules.data_handling import loadmap, checkOption, cbinding, returnBool

class initcondition(object):

    """
    READ/WRITE INITIAL CONDITIONS
    all initial condition can be stored at the end of a run to be used as a **warm** start for a following up run


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    coverTypes            land cover types - forest - grassland - irrPaddy - irrNonPaddy - water - sealed   --       
    loadInit              Flag: if true initial conditions are loaded                                       --       
    initLoadFile          load file name of the initial condition data                                      --       
    saveInit              Flag: if true initial conditions are saved                                        --       
    saveInitFile          save file name of the initial condition data                                      --       
    ====================  ================================================================================  =========

    **Functions**
    """


    def __init__(self, model):
        self.var = model.data.grid
        self.model = model


    def initial(self):
        """
        initial part of the initcondition module
		Puts all the variables which has to be stored in 2 lists:

		* self.initCondVar: the name of the variable in the init netcdf file
		* self.initCondVarValue: the variable as it can be read with the 'eval' command

		Reads the parameter *save_initial* and *save_initial* to know if to save or load initial values
        """

        self.model.init_save_folder = os.path.join(self.model.config['general']['report_folder'], 'init')

        self.initCondVar = ['landunit.w1', 'landunit.w2', 'landunit.w3', 'landunit.topwater', 'landunit.interceptStor', 'landunit.SnowCoverS', 'landunit.FrostIndex', 'grid.channelStorageM3', 'grid.discharge', 'grid.lakeInflow', 'grid.lakeStorage', 'grid.reservoirStorage', 'grid.lakeVolume', 'grid.outLake', 'grid.lakeOutflow']
        
        if returnBool('useSmallLakes'):
            self.initCondVar.extend(['grid.smalllakeInflow', 'grid.smalllakeStorage', 'grid.smalllakeOutflow', 'grid.smalllakeInflowOld', 'grid.smalllakeVolumeM3'])

        self.model.coverTypes = ["forest", "grassland", "irrPaddy", "irrNonPaddy", "sealed", "water"]
        return

        # water demand
        self.initCondVar.append("unmetDemandPaddy")
        self.initCondVarValue.append("unmetDemandPaddy")
        self.initCondVar.append("unmetDemandNonpaddy")
        self.initCondVarValue.append("unmetDemandNonpaddy")

        # groundwater
        self.initCondVar.append("storGroundwater")
        self.initCondVarValue.append("storGroundwater")

    def dynamic(self):
        """
        Dynamic part of the initcondition module
        write initital conditions into a single netcdf file

        Note:
            Several dates can be stored in different netcdf files
        """

        if self.model.save_initial:
            if self.model.n_timesteps == self.model.current_timestep:
                if not os.path.exists(self.model.init_save_folder):
                    os.makedirs(self.model.init_save_folder)
                for initvar in self.initCondVar:
                    fp = os.path.join(self.model.init_save_folder, f"{initvar}.npy")
                    values = attrgetter(initvar)(self.model.data)
                    np.save(fp, values)

