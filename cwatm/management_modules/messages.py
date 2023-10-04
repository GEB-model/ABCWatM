# -------------------------------------------------------------------------
# Name:        Messages
# Purpose:
#
# Author:      burekpe
#
# Created:     16/05/2016
# Copyright:   (c) burekpe 2016
# -------------------------------------------------------------------------


import os
import sys


class CWATMError(Exception):
    """
    The error handling class
    prints out an error

    :param Warning: class CWATMError
    :return: prints out a message about an error
    """

    def __init__(self, msg):

        # don't show the error code, lines etc.
        # sys.tracebacklimit = 0
        header = "\n\n ========================== CWATM ERROR =============================\n"
        try:
           self._msg = header + msg +"\n" +  sys.exc_info()[1].message
           print(self._msg)
        except:
           self._msg = header + msg +"\n"
    def __str__(self):
        return self._msg


class CWATMFileError(CWATMError):
    """
    The error handling class
    prints out an error

    :param Warning: class CWATMError
    :return: prints out a message about file error

    """
    def __init__(self, filename, msg="", sname=""):
        # don't show the error code, lines etc.
        path, name = os.path.split(filename)
        if os.path.exists(filename):
            text1 = "In  \"" + sname + "\"\n"
            text1 += "filename: "+ str(filename) + " exists, but an error was raised"
        elif os.path.exists(path):
            text1 = "In  \"" + sname + "\"\n"
            text1 += "path: "+ path + " exists\nbut filename: "+name+ " does not\n"
            text1 +="file name extension can be .nc4 or .nc\n"
        else:
            text1 = " In  \""+ sname +"\"\n"
            text1 += "searching: \""+filename+"\""
            text1 += "\npath: "+ path + " does not exists\n"

        header = "\n\n ======================== CWATM FILE ERROR ===========================\n"
        self._msg = header + msg + text1

class CWATMWarning(Warning):
    """
    the error handling class
    prints out an error

    :param Warning: class warning
    :return: prints out a message
    """

    def __init__(self, msg):
        sys.tracebacklimit = 0
        header = "\n\n ========================== CWATM Warning =============================\n"
        self._msg = header + msg
        sys.tracebacklimit = 1
    def __str__(self):
        return self._msg

class CWATMRunInfo(Warning):
    """
    prints out an error

    :param Warning: class warning
    :return: prints out a message

    Warning
        warning given with a header and a message from the subroutine
    """

    def __init__(self, outputS):
        header = "\nCWATM Simulation Information and Setting\n"
        msg = "The simulation output as specified in the settings file: " + str(outputS[0]) + " can be found in "+str(outputS[1])+"\n"
        self._msg = header + msg
    def __str__(self):
        return self._msg

