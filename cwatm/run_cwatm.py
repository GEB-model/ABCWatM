#!/usr/bin/env python3.7

"""
::

 -------------------------------------------------
 ######## ##          ##  ####  ######  ##    ##
 ##       ##          ## ##  ##   ##   ####  ####
 ##        ##        ##  ##  ##   ##   ## #### ##
 ##        ##   ##   ## ########  ##  ##   ##   ##
 ##         ## #### ##  ##    ##  ##  ##        ##
 ##         ####  #### ##      ## ## ##          ##
 ##########  ##    ##  ##      ## ## ##          ##

 Community WATer Model


CWATM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

CWATM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details
<http://www.gnu.org/licenses/>.

# --------------------------------------------------
"""

from cwatm import (
    __author__,
    __version__,
    __date__,
    __copyright__,
    __maintainer__,
    __status__,
)

import os
import sys
import datetime

from cwatm.management_modules.globals import (
    globalFlags,
    settingsfile,
    versioning,
    platform1,
    globalclear,
)
from cwatm.management_modules.data_handling import Flags


def usage():
    """
    Prints some lines describing how to use this program which arguments and parameters it accepts, etc

    * -q --quiet       output progression given as .
    * -v --veryquiet   no output progression is given
    * -l --loud        output progression given as time step, date and discharge
    * -c --check       input maps and stack maps are checked, output for each input map BUT no model run
    * -h --noheader    .tss file have no header and start immediately with the time series
    * -t --printtime   the computation time for hydrological modules are printed

    """
    print("CWatM - Community Water Model")
    print("Authors: ", __author__)
    print("Version: ", __version__)
    print("Date: ", __date__)
    print("Status: ", __status__)
    print(
        """
    Arguments list:
    settings.ini     settings file

    -q --quiet       output progression given as .
    -v --veryquiet   no output progression is given
    -l --loud        output progression given as time step, date and discharge
    -c --check       input maps and stack maps are checked, output for each input map BUT no model run
    -h --noheader    .tss file have no header and start immediately with the time series
    -t --printtime   the computation time for hydrological modules are printed
    -w --warranty    copyright and warranty information
    """
    )
    return True


def GNU():
    """
    prints GNU General Public License information

    """

    print("CWatM - Community Water Model")
    print("Authors: ", __author__)
    print("Version: ", __version__)
    print("Date: ", __date__)
    print()
    print(
        """
    CWATM is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details
    <http://www.gnu.org/licenses/>.
    """
    )
    sys.exit(1)


def headerinfo():
    """
    Print the information on top of each run

    this is collecting the last change of one of the source files
    in order to give more information of the settingsfile and the version of cwatm
    this information is put in the result files .tss and .nc
    """

    versioning["exe"] = __file__
    realPath = os.path.dirname(os.path.realpath(versioning["exe"]))
    i = 0
    for dirpath, _, filenames in os.walk(realPath):
        for file in filenames:
            if file[-3:] == ".py":
                i += 1
                file1 = dirpath + "/" + file
                if i == 1:
                    lasttime = os.path.getmtime(file1)
                    lastfile = file
                else:
                    if os.path.getmtime(file1) > lasttime:
                        lasttime = os.path.getmtime(file1)
                        lastfile = file
    versioning["lastdate"] = datetime.datetime.fromtimestamp(lasttime).strftime(
        "%Y/%m/%d %H:%M"
    )
    __date__ = versioning["lastdate"]
    versioning["lastfile"] = lastfile
    versioning["version"] = __version__
    versioning["platform"] = platform1

    if not (Flags["veryquiet"]) and not (Flags["quiet"]):
        print(
            "CWATM - Community Water Model ",
            __version__,
            " Date: ",
            versioning["lastdate"],
            " ",
        )
        print("International Institute of Applied Systems Analysis (IIASA)")
        print("Running under platform: ", platform1)
        print("-----------------------------------------------------------")


def main(settings, args):
    success = False
    if Flags["test"]:
        globalclear()

    globalFlags(settings, args, settingsfile, Flags)
    if Flags["use"]:
        usage()
    if Flags["warranty"]:
        GNU()
    # setting of global flag e.g checking input maps, producing more output information
    headerinfo()
    success, last_dis = CWATMexe(settingsfile[0])

    # if Flags['test']:
    return success, last_dis


def parse_args():
    if len(sys.argv) < 2:
        usage()
        sys.exit(0)
    else:
        return sys.argv[1], sys.argv[2:]


def run_from_command_line():
    settings, args = parse_args()
    main(settings, args)


if __name__ == "__main__":
    settings, args = parse_args()
    main(settings, args)
