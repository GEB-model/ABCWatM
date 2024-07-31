# -------------------------------------------------------------------------
# Name:       CWAT Model Dynamic
# Purpose:
#
# Author:      burekpe
#
# Created:     16/05/2016
# Copyright:   (c) burekpe 2016
# -------------------------------------------------------------------------

from geb.workflows import TimingModule


class CWATModel_dyn:
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

        timer = TimingModule("CWatM")

        self.evaporationPot_module.dynamic()
        timer.new_split("PET")

        self.lakes_reservoirs_module.dynamic()
        timer.new_split("Waterbodies")

        self.snowfrost_module.dynamic()
        timer.new_split("Snow and frost")

        (
            interflow,
            directRunoff,
            groundwater_recharge,
            groundwater_abstraction,
            channel_abstraction,
            openWaterEvap,
            returnFlow,
        ) = self.landcoverType_module.dynamic()
        timer.new_split("Landcover")

        self.groundwater_modflow_module.dynamic(
            groundwater_recharge, groundwater_abstraction
        )
        timer.new_split("GW")

        self.runoff_concentration_module.dynamic(interflow, directRunoff)
        timer.new_split("Runoff concentration")

        self.lakes_res_small_module.dynamic()
        timer.new_split("Small waterbodies")

        self.routing_kinematic_module.dynamic(
            openWaterEvap, channel_abstraction, returnFlow
        )
        timer.new_split("Routing")

        if self.timing:
            print(timer)
