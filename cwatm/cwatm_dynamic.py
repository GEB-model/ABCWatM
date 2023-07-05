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



        if Flags['loud']:
            print("%-6i %10s" %(self.current_timestep, self.current_time_fmt), end=' ')
        else:
            if not(Flags['check']):
                if (Flags['quiet']) and (not(Flags['veryquiet'])):
                    sys.stdout.write(".")
                if (not(Flags['quiet'])) and (not(Flags['veryquiet'])):
                    sys.stdout.write("\r%d" % self.current_timestep)
                    sys.stdout.flush()
                if not (Flags['veryquiet']): print()

        # ************************************************************
        """ up to here it was fun, now the real stuff starts
        """


        if checkOption('calc_environflow') and (returnBool('calc_ef_afterRun')  == False):
            # if only the dis is used for calculation of EF
            self.environflow_module.dynamic()
            # self.output_module.dynamic(ef = True)
            sys.exit("done with Environmental Flow")


        # self.readmeteo_module.dynamic()
        # timemeasure("Read meteo") # 1. timing after read input maps

        self.evaporationPot_module.dynamic()
        timemeasure("ET pot") # 2. timing after read input maps

        #if Flags['check']: return  # if check than finish here

        """ Here it starts with hydrological modules:
        """

        # ***** INFLOW HYDROGRAPHS (OPTIONAL)****************
        self.inflow_module.dynamic()
        self.lakes_reservoirs_module.dynamic()

        # ***** RAIN AND SNOW *****************************************
        self.snowfrost_module.dynamic()
        timemeasure("Snow")  # 3. timing

        # ***** READ land use fraction maps***************************

        # *********  Soil splitted in different land cover fractions *************
        interflow, directRunoff, groundwater_recharge, groundwater_abstraction, channel_abstraction, openWaterEvap, returnFlow = self.landcoverType_module.dynamic()
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