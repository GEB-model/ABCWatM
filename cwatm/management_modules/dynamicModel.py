class DynamicModel:
    i = 1


class ModelFrame:
    """
    Frame of the dynamic hydrological model

    lastTimeStep:  Last time step to run
    firstTimestep: Starting time step of the model
    """

    def __init__(self, model, lastTimeStep=1, firstTimestep=1):
        """
        sets first and last time step into the model

        :param lastTimeStep: last timestep
        :param firstTimeStep: first timestep
        :return: -
        """

        self.model = model

    def step(self):
        self.model.dynamic()

    def finalize(self):
        """
        Finalize the model
        """
        # finalize modflow model
        self.model.groundwater_modflow_module.modflow.finalize()

        if self.model.config["general"]["simulate_forest"]:
            for plantFATE_model in self.model.plantFATE:
                if plantFATE_model is not None:
                    plantFATE_model.finalize()
