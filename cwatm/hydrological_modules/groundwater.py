# -------------------------------------------------------------------------
# Name:        Groundwater module
# Purpose:
#
# Author:      PB
#
# Created:     15/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

from cwatm.management_modules.data_handling import *


class groundwater(object):
    """
    GROUNDWATER


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    modflow               Flag: True if modflow_coupling = True in settings file                            --       
    storGroundwater       simulated groundwater storage                                                     m        
    specificYield         groundwater reservoir parameters (if ModFlow is not used) used to compute ground  m        
    recessionCoeff        groundwater storage times this coefficient gives baseflow                         --       
    kSatAquifer           groundwater reservoir parameters (if ModFlow is not used), could be used to comp  m day-1  
    load_initial                                                                                                     
    readAvlStorGroundwat  same as storGroundwater but equal to 0 when inferior to a treshold                m        
    pumping_actual                                                                                                   
    capillar              Simulated flow from groundwater to the third CWATM soil layer                     m        
    baseflow              simulated baseflow (= groundwater discharge to river)                             m        
    gwstore                                                                                                          
    prestorGroundwater    storGroundwater at the beginning of each step                                     m        
    nonFossilGroundwater  groundwater abstraction which is sustainable and not using fossil resources       m        
    sum_gwRecharge        groundwater recharge                                                              m        
    waterbalance_module                                                                                              
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.grid
        self.model = model
        
    def initial(self):
        """
        Initial part of the groundwater module

        * load parameters from settings file
        * initial groundwater storage
        """

        self.var.recessionCoeff = loadmap('recessionCoeff')

        # for CALIBRATION
        self.var.recessionCoeff = 1 / self.var.recessionCoeff * loadmap('recessionCoeff_factor')
        self.var.recessionCoeff = 1 / self.var.recessionCoeff


        self.var.specificYield = loadmap('specificYield')
        self.var.kSatAquifer = loadmap('kSatAquifer')

        #report("C:/work/output2/ksat.map", self.var.kSatAquifer)

        # init calculation recession coefficient, speciefic yield, ksatAquifer
        self.var.recessionCoeff = np.maximum(5.e-4,self.var.recessionCoeff)
        self.var.recessionCoeff = np.minimum(1.000,self.var.recessionCoeff)
        self.var.specificYield  = np.maximum(0.010,self.var.specificYield)
        self.var.specificYield  = np.minimum(1.000,self.var.specificYield)
        self.var.kSatAquifer = np.maximum(0.010,self.var.kSatAquifer)
        self.var.head_development = globals.inZero.copy()

        # initial conditions
        self.var.storGroundwater = self.var.load_initial('storGroundwater')
        self.var.storGroundwater = np.maximum(3.0, self.var.storGroundwater) + globals.inZero #Establishing initial storGroundwater
        self.var.prestorGroundwater = self.var.storGroundwater.copy()

        # for water demand to have some initial value
        tresholdStorGroundwater = 0.00005  # 0.05 mm
        # self.var.readAvlStorGroundwater = np.where(self.var.storGroundwater > tresholdStorGroundwater, self.var.storGroundwater,0.0)

        # MODFLOW initialization
        self.var.pumping_actual = self.var.full_compressed(0, dtype=np.float32)
        self.var.capillar = self.var.full_compressed(0, dtype=np.float32)
        self.var.baseflow = self.var.full_compressed(0, dtype=np.float32)
        self.var.gwstore = self.var.full_compressed(0, dtype=np.float32)

    def dynamic(self):
        """
        Dynamic part of the groundwater module
        Calculate groundweater storage and baseflow
        """

        #self.var.sum_gwRecharge = readnetcdf2("C:/work/output2/sum_gwRecharge_daily.nc", dateVar['currDate'], addZeros=True, cut = False, usefilename = True )

        # WATER DEMAND
        # update storGoundwater after self.var.nonFossilGroundwaterAbs
        self.var.storGroundwater = np.maximum(0., self.var.storGroundwater - self.model.data.to_grid(landunit_data=self.model.data.landunit.nonFossilGroundwaterAbs, fn='mean'))
        # PS: We assume only local groundwater abstraction can happen (only to satisfy water demand within a cell).
        # unmetDemand (m), satisfied by fossil gwAbstractions (and/or desalinization or other sources)
        # (equal to zero if limitAbstraction = True)

        # get riverbed infiltration from the previous time step (from routing)
        #self.var.surfaceWaterInf = self.var.riverbedExchange * self.var.InvCellArea
        #self.var.storGroundwater = self.var.storGroundwater + self.var.surfaceWaterInf

        # get net recharge (percolation-capRise) and update storage:
        self.var.storGroundwater = self.var.storGroundwater + self.var.sum_gwRecharge

        self.var.storGroundwater = self.var.storGroundwater - self.var.baseflow - self.var.capillar
        assert (self.var.storGroundwater > 0).all()

        # PS: baseflow must be calculated at the end (to ensure the availability of storGroundwater to support nonFossilGroundwaterAbs)

        # to avoid small values and to avoid excessive abstractions from dry groundwater
        tresholdStorGroundwater = 0.00001  # 0.01 mm
        self.var.readAvlStorGroundwater = np.where(
            self.var.storGroundwater > tresholdStorGroundwater,
            self.var.storGroundwater - tresholdStorGroundwater,
            0
        )

