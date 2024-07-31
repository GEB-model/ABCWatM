# -------------------------------------------------------------------------
# Name:        Water Balance module
# Purpose:
# 1.) check if water balance per time step is ok ( = 0)
# 2.) produce an annual overview - income, outcome storage
# Author:      PB
#
# Created:     22/08/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------
import numpy as np
from cwatm.data_handling import checkOption


class waterbalance(object):
    """
    WATER BALANCE
    """

    def __init__(self, model):
        self.var = model.data.grid
        self.model = model

    def initial(self):
        """
        Initial part of the water balance module
        """

    def waterBalanceCheck(
        self,
        name=None,
        how="cellwise",
        influxes=[],
        outfluxes=[],
        prestorages=[],
        poststorages=[],
        tollerance=1e-10,
    ):
        """
        Dynamic part of the water balance module

        Returns the water balance for a list of input, output, and storage map files

        :param influxes: income
        :param outfluxes: this goes out
        :param prestorages:  this was in before
        :param endStorages:  this was in afterwards
        :return: -
        """

        income = 0
        out = 0
        store = 0

        assert isinstance(influxes, list)
        assert isinstance(outfluxes, list)
        assert isinstance(prestorages, list)
        assert isinstance(poststorages, list)

        if how == "cellwise":
            for fluxIn in influxes:
                income += fluxIn
            for fluxOut in outfluxes:
                out += fluxOut
            for preStorage in prestorages:
                store += preStorage
            for endStorage in poststorages:
                store -= endStorage
            balance = income + store - out

            if balance.size == 0:
                return True
            elif np.abs(balance).max() > tollerance:
                text = f"{balance[np.abs(balance).argmax()]} is larger than tollerance {tollerance}"
                if name:
                    print(name, text)
                else:
                    print(text)
                # raise AssertionError(text)
                return False
            else:
                return True

        elif how == "sum":
            for fluxIn in influxes:
                income += fluxIn.sum()
            for fluxOut in outfluxes:
                out += fluxOut.sum()
            for preStorage in prestorages:
                store += preStorage.sum()
            for endStorage in poststorages:
                store -= endStorage.sum()

            balance = income + store - out
            if balance > tollerance:
                text = f"{np.abs(balance).max()} is larger than tollerance {tollerance}"
                print(text)
                if name:
                    print(name, text)
                else:
                    print(text)
                # raise AssertionError(text)
                return False
            else:
                return True
        else:
            raise ValueError(f"Method {how} not recognized.")
