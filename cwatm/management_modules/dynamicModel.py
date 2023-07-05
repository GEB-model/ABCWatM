# initial and dynamic model-> idea taken from PC_Raster

class DynamicModel:
    i = 1

# ----------------------------------------

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
