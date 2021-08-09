# -------------------------------------------------------------------------
# Name:        Waterdemand modules
# Purpose:
#
# Author:      PB, YS, MS, JdB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules import globals
from cwatm.management_modules.data_handling import returnBool, binding, readnetcdf2
import numpy as np

class waterdemand_environmental_need:
    """
    WATERDEMAND environment_need

    calculating water demand -
    environmental need based on precalculated maps done before in CWatM

    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    cut_ef_map            if TRUE calculated maps of environmental flow are cut to the extend of the area   --       
    M3toM                 Coefficient to change units                                                       --       
    chanLength                                                                                                       
    channelAlpha                                                                                                     
    use_environflow                                                                                                  
    envFlowm3s                                                                                                       
    envFlow                                                                                                          
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.subvar
        self.model = model

    def initial(self):
        pass

    def dynamic(self):
        return self.var.full_compressed(0, dtype=np.float32)