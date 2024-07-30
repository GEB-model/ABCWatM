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

import sys


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


def parse_args():
    if len(sys.argv) < 2:
        usage()
        sys.exit(0)
    else:
        return sys.argv[1], sys.argv[2:]
