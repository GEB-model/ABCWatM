import numpy as np
from .cwatm_initial import CWATModel_ini
from .cwatm_dynamic import CWATModel_dyn


class CWATModel(CWATModel_ini, CWATModel_dyn):
    """
    Initial and dynamic part of the CWATM model
    * initial part takes care of all the non temporal initialiation procedures
    * dynamic part loops over time
    """

    @property
    def n_individuals_per_m2(self):
        n_invidiuals_per_m2_per_HRU = np.array(
            [model.n_individuals for model in self.plantFATE if model is not None]
        )
        land_use_ratios = self.data.HRU.land_use_ratio[
            self.soil_module.plantFATE_forest_RUs
        ]
        return np.array(
            (n_invidiuals_per_m2_per_HRU * land_use_ratios).sum()
            / land_use_ratios.sum()
        )

    @property
    def biomass_per_m2(self):
        biomass_per_m2_per_HRU = np.array(
            [model.biomass for model in self.plantFATE if model is not None]
        )
        land_use_ratios = self.data.HRU.land_use_ratio[
            self.soil_module.plantFATE_forest_RUs
        ]
        return np.array(
            (biomass_per_m2_per_HRU * land_use_ratios).sum() / land_use_ratios.sum()
        )
