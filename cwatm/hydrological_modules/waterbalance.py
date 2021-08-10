#-------------------------------------------------------------------------
#Name:        Water Balance module
#Purpose:
#1.) check if water balance per time step is ok ( = 0)
#2.) produce an annual overview - income, outcome storage
#Author:      PB
#
#Created:     22/08/2016
#Copyright:   (c) PB 2016
#-------------------------------------------------------------------------

from cwatm.management_modules.data_handling import *

class waterbalance(object):
    """
    WATER BALANCE
    
    * check if water balnace per time step is ok ( = 0)
    * produce an annual overview - income, outcome storage


    **Global variables**

    ====================  ================================================================================  =========
    Variable [self.var]   Description                                                                       Unit     
    ====================  ================================================================================  =========
    nonIrrReturnFlow                                                                                                 
    localQW                                                                                                          
    channelStorageM3Before                                                                                             
    sum_balanceStore                                                                                                 
    sum_balanceFlux                                                                                                  
    catchmentAll                                                                                                     
    ====================  ================================================================================  =========

    **Functions**
    """

    def __init__(self, model):
        self.var = model.data.var
        self.model = model

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

    def initial(self):
        """
        Initial part of the water balance module
        """

        if checkOption('calcWaterBalance'):

            self.var.nonIrrReturnFlow = 0
            self.var.localQW = 0
            self.var.channelStorageM3Before = 0

        """ store the initial storage volume of snow, soil etc.
        """
        if checkOption('sumWaterBalance'):
            # variables of storage
            self.var.sum_balanceStore = ['SnowCover','sum_interceptStor','sum_topWaterLayer']

            # variable of fluxes
            self.var.sum_balanceFlux = ['Precipitation','SnowMelt','Rain','sum_interceptEvap','actualET']

            #for variable in self.var.sum_balanceStore:
                # vars(self.var)["sumup_" + variable] =  vars(self.var)[variable]
            for variable in self.var.sum_balanceFlux:
                vars(self.var)["sumup_" + variable] =  globals.inZero.copy()



# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

    def waterBalanceCheck(self, name=None, how='cellwise', influxes=[], outfluxes=[], prestorages=[], poststorages=[], tollerance=1e-10):
        """
        Dynamic part of the water balance module

        Returns the water balance for a list of input, output, and storage map files

        :param influxes: income
        :param outfluxes: this goes out
        :param prestorages:  this was in before
        :param endStorages:  this was in afterwards
        :return: -
        """

        income =  0
        out = 0
        store = 0

        assert isinstance(influxes, list)
        assert isinstance(outfluxes, list)
        assert isinstance(prestorages, list)
        assert isinstance(poststorages, list)

        if how == 'cellwise':
            for fluxIn in influxes:   
                income += fluxIn
            for fluxOut in outfluxes:
                out += fluxOut
            for preStorage in prestorages:
                store += preStorage
            for endStorage in poststorages: 
                store -= endStorage
            balance = income + store - out
            
            if np.abs(balance).max() > tollerance:
                text = f"{balance[np.abs(balance).argmax()]} is larger than tollerance {tollerance}"
                if name:
                    print(name, text)
                else:
                    print(text)
                # raise AssertionError(text)
                return False
            else:
                return True
        
        elif how == 'sum':
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

    def waterBalanceCheckSum(self, fluxesIn, fluxesOut, preStorages, endStorages, processName, printTrue=False):
        """
        Returns the water balance for a list of input, output, and storage map files
        and sums it up for a catchment

        :param fluxesIn: income
        :param fluxesOut: this goes out
        :param preStorages:  this was in before
        :param endStorages:  this was in afterwards
        :param processName:  name of the process
        :param printTrue: calculate it?
        :return: Water balance as output on the screen
        """

        if printTrue:
            minB =0
            maxB = 0
            maxBB = 0

            income =  0
            out = 0
            store =  0

            for fluxIn in fluxesIn:
                if not(isinstance(fluxIn,np.ndarray)) : fluxIn = globals.inZero
                income = income + np.bincount(self.var.catchmentAll,weights=fluxIn)
            for fluxOut in fluxesOut:
                if not (isinstance(fluxOut, np.ndarray)): fluxOut = globals.inZero
                out = out + np.bincount(self.var.catchmentAll,weights=fluxOut)
            for preStorage in preStorages:
                if not (isinstance(preStorage, np.ndarray)): preStorage = globals.inZero
                store = store + np.bincount(self.var.catchmentAll,weights=preStorage)
            for endStorage in endStorages:
                if not (isinstance(endStorage, np.ndarray)): endStorage = globals.inZero
                store = store - np.bincount(self.var.catchmentAll,weights=endStorage)
            balance =  income + store - out
            #balance = endStorages
            #if balance.size:
                #minB = np.amin(balance)
                #maxB = np.amax(balance)
                #maxBB = np.maximum(np.abs(minB),np.abs(maxB))
                #meanB = np.average(balance, axis=0)
                #meanB = 0.0
            #no = self.var.catchmentNo
            no = 0


            #print "     %s %10.8f " % (processName, maxBB),
            #print "     %s %10.8f %10.8f" % (processName, minB,maxB),
            #print "     %s %10.8f" % (processName, balance[no]),
            print("     %s %10.8f" % (processName, balance[no]), end=' ')

            #avgArea = npareaaverage(self.var.cellArea, self.var.catchmentAll)
            #dis = balance[no] * avgArea[0] / self.model.DtSec
            #print "     %s %10.8f" % (processName, dis),
            return balance[no]


    def dynamic(self):
        """
        Dynamic part of the water balance module
        If option **sumWaterBalance** sum water balance for certain variables
        """

        #if checkOption('sumWaterBalance'):
        #    i = 1

        # sum up storage variables
        #for variable in self.var.sum_balanceStore:
         #   vars(self.var)["sumup_" + variable] =  vars(self.var)[variable].copy()


        # sum up fluxes variables
        for variable in self.var.sum_balanceFlux:
            vars(self.var)["sumup_" + variable] = vars(self.var)["sumup_" + variable] + vars(self.var)[variable]