# -------------------------------------------------------------------------
# Name:        Handling of timesteps and dates
# Purpose:
#
# Author:      P. Burek
#
# Created:     09/08/2016
# Copyright:   (c) burekpe 2016
# -------------------------------------------------------------------------


import os
import datetime
import time as xtime
import numpy as np
from cwatm.management_modules.data_handling import *
from cwatm.management_modules.globals import *
from cwatm.management_modules.messages import *
from netCDF4 import Dataset, num2date, date2num, date2index

import difflib  # to check the closest word in settingsfile, if an error occurs


def datenum(date):
    """
    converts date to a int number based on the calender and unit of the netcdf file
    :param date:
    :return: number of the date
    """
    num = date2num(date, units=dateVar["unit"], calendar=dateVar["calendar"])
    return num // dateVar["unitConv"]


def numdate(num, add=0):
    """
    converts int into date based on the calender and unit of the netcdf file
    :param num:  number of the day
    :param add:  addition to date in days
    :return: date
    """
    return num2date(
        int(num) * dateVar["unitConv"] + add,
        units=dateVar["unit"],
        calendar=dateVar["calendar"],
    )


def date2str(date):
    """
    Convert date to string of date e.g. 27/12/2018
    :param date: date as (datetime)
    :return: date string
    """

    return "%02d/%02d/%02d" % (date.day, date.month, date.year)


def ctbinding(inBinding):
    """
    Check if variable in settings file has a counterpart in source code

    :param x: variable in settings file to be tested
    :return: -
    :raises: if variable is not found send an error: :meth:`management_modules.messages.CWATMError`
    """

    test = inBinding in binding
    if test:
        return binding[inBinding]
    else:
        closest = difflib.get_close_matches(inBinding, list(binding.keys()))
        if not closest:
            closest = ["- no match -"]
        msg = "===== Timing in the section: [TIME-RELATED_CONSTANTS] is wrong! =====\n"
        msg += (
            'No key with the name: "'
            + inBinding
            + '" in the settings file: "'
            + settingsfile[0]
            + '"\n'
        )
        msg += 'Closest key to the required one is: "' + closest[0] + '"'
        raise CWATMError(msg)


def timemeasure(name, loops=0, update=False, sample=1):
    """
    Measuring of the time for each subroutine

    :param name: name of the subroutine
    :param loops: if it it called several times this is added to the name
    :param update:
    :param sample:
    :return: add a string to the time measure string: timeMesString
    """

    timeMes.append(xtime.perf_counter())
    if loops == 0:
        s = name
    else:
        s = name + "_%i" % loops
    timeMesString.append(s)
    return


# -----------------------------------------------------------------------
# Calendar routines
# -----------------------------------------------------------------------


def Calendar(input, errorNo=0):
    """
    Get the date from CalendarDayStart in the settings xml
    Reformatting the date till it fits to datetime

    :param input: string from the settingsfile should be somehow a date
    :param errorNo: 0: check startdate, enddate 1: check startinit
    :return: a datetime date
    """

    date = None

    try:
        date = float(input)
    except ValueError:
        d = input.replace(".", "/")
        d = d.replace("-", "/")
        year = d.split("/")[-1:]
        if len(year[0]) == 4:
            formatstr = "%d/%m/%Y"
        else:
            formatstr = "%d/%m/%y"
        if len(year[0]) == 1:
            d = d.replace("/", ".", 1)
            d = d.replace("/", "/0")
            d = d.replace(".", "/")
            print(d)

        try:
            date = datetime.datetime.strptime(d, formatstr)
        except:
            if errorNo == 0:
                msg = "Either date in StepStart is not a date or in SpinUp or StepEnd it is neither a number or a date!"
                raise CWATMError(msg)
            elif errorNo == 1:
                msg = "First date in StepInit is neither a number or a date!"
                raise CWATMError(msg)
            elif errorNo > 1:
                return -99999
    return date


def datetoInt(dateIn, begin, both=False):
    """
    Calculates the integer of a date from a reference date

    :param dateIn: date
    :param begin: reference date
    :param both: if set to True both the int and the string of the date are returned
    :return: integer value of a date, starting from begin date
    """

    date1 = Calendar(dateIn)

    if type(date1) is datetime.datetime:
        # str1 = date1.strftime("%d/%m/%Y")
        # to cope with dates before 1990
        str1 = date2str(date1)
        d1 = datenum(date1)
        d2 = datenum(begin)
        int1 = int(d1 - d2) + 1
        # to be used with different days in a year e.g. 360_years

    else:
        int1 = int(date1)
        str1 = str(date1)
    if both:
        return int1, str1
    else:
        return int1


def addmonths(d, x):
    """
    Adds months to a date

    :param d: date
    :param x: month to add
    :return: date with added months
    """

    days_of_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    newmonth = (((d.month - 1) + x) % 12) + 1
    newyear = d.year + (((d.month - 1) + x) // 12)
    if d.day > days_of_month[newmonth - 1]:
        newday = days_of_month[newmonth - 1]
    else:
        newday = d.day
    return datetime.datetime(newyear, newmonth, newday)


# noinspection PyTypeChecker
def checkifDate(start, end, spinup, name):
    """
    Checks if start date is earlier than end date etc
    And set some date variables

    :param start: start date
    :param end: end date
    :param spinup: date till no output is generated = warming up time
    :return: a list of date variable in: dateVar
    """

    # begin = Calendar(ctbinding('CalendarDayStart'))
    try:
        name = glob.glob(os.path.normpath(name))[0]
    except:
        raise CWATMFileError(name, sname="PrecipitationMaps")

    nf1 = Dataset(name, "r")
    try:
        dateVar["calendar"] = nf1.variables["time"].calendar
        dateVar["unit"] = nf1.variables["time"].units
    except:
        dateVar["calendar"] = "standard"
        dateVar["unit"] = "days since 1901-01-01T00:00:00Z"
    nf1.close()

    unitconv1 = ["DAYS", "HOUR", "MINU", "SECO"]
    unitconv2 = [1, 24, 1440, 86400]
    unitconv3 = dateVar["unit"][:4].upper()
    try:
        dateVar["unitConv"] = unitconv2[unitconv1.index(unitconv3)]
    except:
        dateVar["unitConv"] = 1

    startdate = Calendar(ctbinding("StepStart"))
    if type(startdate) is datetime.datetime:
        begin = startdate
    else:
        msg = '"StepStart = ' + ctbinding("StepStart") + '"\n'
        msg += "StepStart has to be a valid date!"
        raise CWATMError(msg)

    # spinup date = date from which maps are written
    if ctbinding(spinup).lower() == "none" or ctbinding(spinup) == "0":
        spinup = start

    dateVar["intStart"], strStart = datetoInt(ctbinding(start), begin, True)
    dateVar["intEnd"], strEnd = datetoInt(ctbinding(end), begin, True)
    dateVar["intSpin"], strSpin = datetoInt(ctbinding(spinup), begin, True)

    # test if start and end > begin
    if (
        (dateVar["intStart"] < 0)
        or (dateVar["intEnd"] < 0)
        or ((dateVar["intEnd"] - dateVar["intStart"]) < 0)
    ):
        # strBegin = begin.strftime("%d/%m/%Y")
        strBegin = date2str(begin)
        msg = (
            "Start Date: "
            + strStart
            + " and/or end date: "
            + strEnd
            + " are wrong!\n or smaller than the first time step date: "
            + strBegin
        )
        raise CWATMError(msg)

    if (dateVar["intSpin"] < dateVar["intStart"]) or (
        dateVar["intSpin"] > dateVar["intEnd"]
    ):
        # strBegin = begin.strftime("%d/%m/%Y")
        strBegin = date2str(begin)
        msg = (
            "Spin Date: "
            + strSpin
            + " is wrong!\n or smaller/bigger than the first/last time step date: "
            + strBegin
            + " - "
            + strEnd
        )
        raise CWATMError(msg)

    dateVar["currDate"] = begin
    dateVar["dateBegin"] = begin
    # dateVar['dateStart'] = begin + datetime.timedelta(days=dateVar['intSpin']-1)

    d1 = datenum(begin)
    startint = int(d1 + dateVar["intSpin"] - 1)
    dateVar["dateStart"] = numdate(startint)

    dateVar["diffdays"] = dateVar["intEnd"] - dateVar["intSpin"] + 1
    # dateVar['dateEnd'] = dateVar['dateStart'] + datetime.timedelta(days=dateVar['diffdays']-1)
    d1 = datenum(dateVar["dateStart"])
    endint = int(d1 + dateVar["diffdays"])
    dateVar["dateEnd"] = numdate(endint, -1)

    dateVar["curr"] = 0
    dateVar["currwrite"] = 0

    # dateVar['datelastmonth'] = datetime.datetime(year=dateVar['dateEnd'].year, month= dateVar['dateEnd'].month, day=1) - datetime.timedelta(days=1)
    d1 = datenum(
        datetime.datetime(
            year=dateVar["dateEnd"].year, month=dateVar["dateEnd"].month, day=1
        )
    )
    dateVar["datelastmonth"] = numdate(d1, -1)
    # dateVar['datelastyear'] = datetime.datetime(year=dateVar['dateEnd'].year, month= 1, day=1) - datetime.timedelta(days=1)
    d1 = datenum(datetime.datetime(year=dateVar["dateEnd"].year, month=1, day=1))
    dateVar["datelastyear"] = numdate(d1, -1)

    dateVar["checked"] = []
    # noinspection PyTypeChecker
    # dates = np.arange(dateVar['dateStart'], dateVar['dateEnd']+ datetime.timedelta(days=1), datetime.timedelta(days = 1)).astype(datetime.datetime)
    # for d in dates:

    for dint in range(startint, endint):
        d = numdate(dint)
        dnext = numdate(dint, 1)
        # if d.day == calendar.monthrange(d.year, d.month)[1]:
        if d.month != dnext.month:
            if d.month == 12:
                dateVar["checked"].append(2)
            else:
                dateVar["checked"].append(1)
        else:
            dateVar["checked"].append(0)

    dateVar["diffMonth"] = dateVar["checked"].count(1) + dateVar["checked"].count(2)
    dateVar["diffYear"] = dateVar["checked"].count(2)


def date2indexNew(date, nctime, calendar, select="nearest", name=""):
    """
    The original netCDF4 library cannot handle month and years
    Replace: date2index
    This one checks for days, month and years
    And set some date variables

    :param date: date
    :param nctime: time unit of the netcdf file
    :param select: (optional) which date is selected, default: nearest
    :param name: (optional) name of th dataset
    :return: index
    """

    unit = nctime.units.split()
    if unit[0].upper() == "DAYS":
        index = date2index(date, nctime, calendar=nctime.calendar, select="nearest")
    elif unit[0][0:5].upper() == "MONTH":
        year0 = int(unit[2][0:4])
        month0 = int(unit[2][6:7])
        value = (date.year - year0) * 12 + (date.month - month0)
        if value > max(nctime[:]):
            value = max(nctime[:]) - 11 + (date.month - month0)
            msg = (
                " - "
                + date.strftime("%Y-%m")
                + " is later then the last dataset in "
                + name
                + " -"
            )
            msg += " instead last year/month dataset is used"
            print(CWATMWarning(msg))

        index = np.where(nctime[:] == value)[0][0]
    elif unit[0][0:4].upper() == "YEAR":
        year0 = int(unit[2][0:4])
        value = date.year - year0
        if value > max(nctime[:]):
            value = max(nctime[:])
            msg = (
                " - "
                + date.strftime("%Y")
                + " is later then the last dataset in "
                + name
                + " -"
            )
            msg += " instead last year dataset is used"
            print(CWATMWarning(msg))

        index = np.where(nctime[:] == value)[0][0]
    else:
        index = date2index(date, nctime, calendar=nctime.calendar, select="nearest")
    return index


def timestep_dynamic(self):
    """
    Dynamic part of setting the date
    Current date is increasing, checking if beginning of month, year

    :return: a list of date variable in: dateVar
    """

    # print "leap:", globals.leap_flag[0]
    # dateVar['currDate'] = dateVar['dateBegin'] + datetime.timedelta(days=dateVar['curr'])
    d1 = datenum(dateVar["dateBegin"])
    dateVar["currDate"] = numdate(d1, dateVar["curr"])
    datevarInt = d1 + dateVar["curr"]

    # dateVar['currDatestr'] = dateVar['currDate'].strftime("%d/%m/%Y")
    dateVar["currDatestr"] = date2str(dateVar["currDate"])

    # dateVar['doy'] = int(dateVar['currDate'].strftime('%j'))
    # replacing this because date less than 1900 is not used
    firstdoy = datetime.datetime(dateVar["currDate"].year, 1, 1)
    # dateVar['doy'] = (dateVar['currDate'] - firstdoy).days + 1
    firstdoyInt = datenum(firstdoy)
    dateVar["doy"] = int(datevarInt - firstdoyInt + 1)
    dateVar["10day"] = int((dateVar["doy"] - 1) / 10)

    dateVar["laststep"] = False
    if (dateVar["intStart"] + dateVar["curr"]) == dateVar["intEnd"]:
        dateVar["laststep"] = True

    dateVar["currStart"] = dateVar["curr"] + 1
    dateVar["curr"] += 1
    # count currwrite only after spin time
    if dateVar["curr"] >= dateVar["intSpin"]:
        dateVar["currwrite"] += 1

    dateVar["currMonth"] = dateVar["checked"][: dateVar["currwrite"]].count(
        1
    ) + dateVar["checked"][: dateVar["currwrite"]].count(2)
    dateVar["currYear"] = dateVar["checked"][: dateVar["currwrite"]].count(2)

    # first timestep
    dateVar["newStart"] = dateVar["curr"] == 1
    dateVar["newMonth"] = dateVar["currDate"].day == 1
    dateVar["newYear"] = (dateVar["currDate"].day == 1) and (
        dateVar["currDate"].month == 1
    )
    dateVar["new10day"] = ((dateVar["doy"] - 1) / 10.0) == dateVar["10day"]

    d1month = datenum(
        datetime.datetime(
            year=dateVar["currDate"].year, month=dateVar["currDate"].month, day=1
        )
    )
    if dateVar["currDate"].month == 12:
        month = 1
        year = dateVar["currDate"].year + 1
    else:
        month = dateVar["currDate"].month + 1
        year = dateVar["currDate"].year
    d2month = datenum(datetime.datetime(year=year, month=month, day=1))

    d1year = datenum(datetime.datetime(year=dateVar["currDate"].year, month=1, day=1))
    d2year = datenum(
        datetime.datetime(year=dateVar["currDate"].year + 1, month=1, day=1)
    )
    dateVar["daysInMonth"] = d2month - d1month
    dateVar["daysInYear"] = d2year - d1year

    return
