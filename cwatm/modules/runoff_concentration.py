# -------------------------------------------------------------------------
# Name:        runoff concentration module
# Purpose:	   this is the part between runoff generation and routing
#              for each gridcell and for each land cover class the generated runoff is concentrated at a corner of a gridcell
#              this concentration needs some lag-time (and peak time) and leads to diffusion
#              lag-time/ peak time is calculated using slope, length and land cover class
#              diffusion is calculated using a triangular-weighting-function
# Author:      PB
#
# Created:     16/12/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------


class runoff_concentration(object):
    """
    Runoff concentration

    this is the part between runoff generation and routing
    for each gridcell and for each land cover class the generated runoff is concentrated at a corner of a gridcell
    this concentration needs some lag-time (and peak time) and leads to diffusion
    lag-time/ peak time is calculated using slope, length and land cover class
    diffusion is calculated using a triangular-weighting-function

    :math:`Q(t) = \sum_{i=0}^{max} c(i) * Q_{\mathrm{GW}} (t - i + 1)`

    where :math:`c(i) = \int_{i-1}^{i} {2 \over{max}} - | u - {max \over {2}} | * {4 \over{max^2}} du`

    see also:

    http://stackoverflow.com/questions/24040984/transformation-using-triangular-weighting-function-in-python







    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit
    ====================  ================================================================================  =========
    load_initial
    baseflow              simulated baseflow (= groundwater discharge to river)                             m
    coverTypes            land cover types - forest - grassland - irrPaddy - irrNonPaddy - water - sealed   --
    runoff
    fracVegCover          Fraction of area covered by the corresponding landcover type
    sum_interflow
    runoff_peak           peak time of runoff in seconds for each land use class                            s
    tpeak_interflow       peak time of interflow                                                            s
    tpeak_baseflow        peak time of baseflow                                                             s
    maxtime_runoff_conc   maximum time till all flow is at the outlet                                       s
    runoff_conc           runoff after concentration - triangular-weighting method                          m
    sum_landSurfaceRunof  Runoff concentration above the soil more interflow including all landcover types  m
    landSurfaceRunoff     Runoff concentration above the soil more interflow                                m
    directRunoff          Simulated surface runoff                                                          m
    interflow             Simulated flow reaching runoff instead of groundwater                             m
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        """
        Initial part of the  runoff concentration module

        Setting the peak time for:

        * surface runoff = 3
        * interflow = 4
        * baseflow = 5

        based on the slope the concentration time for each land cover type is calculated

        Note:
            only if option **includeRunoffConcentration** is TRUE
        """
        self.var = model.data.grid
        self.model = model

    def dynamic(self, interflow, directRunoff):
        self.var.runoff = directRunoff + interflow + self.var.baseflow
