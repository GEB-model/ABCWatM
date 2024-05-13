# -------------------------------------------------------------------------
# Name:       CWAT Model Dynamic
# Purpose:
#
# Author:      burekpe
#
# Created:     16/05/2016
# Copyright:   (c) burekpe 2016
# -------------------------------------------------------------------------

from cwatm.management_modules.data_handling import *
from cwatm.management_modules.messages import *

class CWATModel_dyn(DynamicModel):

    # =========== DYNAMIC ====================================================

    def dynamic(self):
        """
        Dynamic part of CWATM
        calls the dynamic part of the hydrological modules
        Looping through time and space

        Note:
            if flags set the output on the screen can be changed e.g.

            * v: no output at all
            * l: time and first gauge discharge
            * t: timing of different processes at the end
        """

        #self.CalendarDate = dateVar['dateStart'] + datetime.timedelta(days=dateVar['curr'])
        #self.CalendarDay = int(self.CalendarDate.strftime("%j"))
        # timestep_dynamic(self)


        del timeMes[:]
        timemeasure("Start dynamic")

        if checkOption('calc_environflow') and (returnBool('calc_ef_afterRun')  == False):
            # if only the dis is used for calculation of EF
            self.environflow_module.dynamic()
            # self.output_module.dynamic(ef = True)
            sys.exit("done with Environmental Flow")


        # self.readmeteo_module.dynamic()
        # timemeasure("Read meteo") # 1. timing after read input maps

        self.ETref_forest, self.ETref_agriculture, self.ETref_grassland, self.averagetemp_forest, self.averagetemp_agriculture, self.averagetemp_grassland, self.humidity_forest,self.humidity_agriculture,self.humidity_grassland= self.evaporationPot_module.dynamic()
        timemeasure("ET pot") # 2. timing after read input maps

        #if Flags['check']: return  # if check than finish here

        """ Here it starts with hydrological modules:
        """

        # ***** INFLOW HYDROGRAPHS (OPTIONAL)****************
        self.inflow_module.dynamic()
        self.lakes_reservoirs_module.dynamic()

        # ***** RAIN AND SNOW *****************************************
        self.snowmelt = self.snowfrost_module.dynamic()
        timemeasure("Snow")  # 3. timing

        # ***** READ land use fraction maps***************************

        # *********  Soil splitted in different land cover fractions *************
        interflow, directRunoff, groundwater_recharge, groundwater_abstraction, channel_abstraction, openWaterEvap, returnFlow, self.et_forest, self.et_grassland, self.et_agriculture, self.interceptcap_forest, self.interceptcap_grassland, self.interceptcap_agriculture, self.interceptevap_forest, self.interceptevap_grassland,  self.interceptevap_agriculture, self.potET_forest, self.potET_grassland,  self.potET_agriculture, self.soilwaterstorage_forest,self.soilwaterstorage_grassland, self.soilwaterstorage_agriculture,self.infiltration_forest,self.infiltration_grassland,self.infiltration_agriculture,self.potentialinfiltration_forest,self.potentialinfiltration_grassland,self.potentialinfiltration_agriculture, self.rain_forest,  self.rain_agriculture,  self.rain_grassland, self.transpiration_decid, self.transpiration_conifer, self.transpiration_mixed, self.percolation_forest, self.percolation_agriculture, self.percolation_grassland, self.interflow_forest, self.interflow_agriculture, self.interflow_grassland, self.directrunoff_forest, self.directrunoff_agriculture, self.directrunoff_grassland, self.baresoil_forest,self.baresoil_agriculture, self.baresoil_grassland = self.landcoverType_module.dynamic()
        timemeasure("Soil main")  # 5. timing

        self.groundwater_modflow_module.dynamic(groundwater_recharge, groundwater_abstraction)
        timemeasure("Groundwater")  # 7. timing

        self.runoff_concentration_module.dynamic(interflow, directRunoff)
        timemeasure("Runoff conc.")  # 8. timing

        self.lakes_res_small_module.dynamic()
        timemeasure("Small lakes")  # 9. timing
        


        self.routing_kinematic_module.dynamic(openWaterEvap, channel_abstraction, returnFlow)
        timemeasure("Routing_Kin")  # 10. timing

        self.waterquality1.dynamic()
        # *******  Calculate CUMULATIVE MASS BALANCE ERROR  **********
        # self.waterbalance_module.dynamic()

        # ------------------------------------------------------
        # End of calculation -----------------------------------
        # ------------------------------------------------------

        # self.waterbalance_module.checkWaterSoilGround()
        timemeasure("Waterbalance")  # 11. timing

        self.environflow_module.dynamic()
        # in case environmental flow is calculated last