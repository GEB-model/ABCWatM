# -------------------------------------------------------------------------
# Name:        Data handling
# Purpose:     Transforming netcdf to numpy arrays, checking mask file
#
# Author:      PB
#
# Created:     13/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import os, glob

import numpy as np
import math
from numba import njit

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np
from typing import Tuple


from . import globals
from cwatm.management_modules.globals import (
    binding,
    geotrans,
    maskmapAttr,
    maskinfo,
    Flags,
    metaNetcdfVar,
    option,
)

from cwatm.management_modules.dynamicModel import *

from netCDF4 import Dataset, date2num

from osgeo import osr, gdal, gdalconst
import warnings


def reduce_precision(data):
    if isinstance(data, (np.ndarray, cp.ndarray)) and data.dtype == np.float64:
        data = data.astype(np.float32)
    return data


def valuecell(coordx, coordstr, returnmap=True):
    """
    to put a value into a raster map -> invert of cellvalue, map is converted into a numpy array first

    :param coordx: x,y or lon/lat coordinate
    :param coordstr: String of coordinates
    :return: 1D array with new value
    """

    coord = []
    col = []
    row = []
    for xy in coordx:
        try:
            coord.append(float(xy))
        except:
            msg = "Gauges: " + xy + " in " + coordstr + " is not a coordinate"
            raise Exception(msg)

    null = np.zeros((maskmapAttr["row"], maskmapAttr["col"]))
    null[null == 0] = -9999

    for i in range(int(len(coord) / 2)):
        col.append(int((coord[i * 2] - maskmapAttr["x"]) * maskmapAttr["invcell"]))
        row.append(int((maskmapAttr["y"] - coord[i * 2 + 1]) * maskmapAttr["invcell"]))

        if (
            col[i] >= 0
            and row[i] >= 0
            and col[i] < maskmapAttr["col"]
            and row[i] < maskmapAttr["row"]
        ):
            null[row[i], col[i]] = i + 1
        else:
            x1 = maskmapAttr["x"]
            x2 = x1 + maskmapAttr["cell"] * maskmapAttr["col"]
            y1 = maskmapAttr["y"]
            y2 = y1 - maskmapAttr["cell"] * maskmapAttr["row"]
            box = "%5s %5.1f\n" % ("", y1)
            box += "%5s ---------\n" % ""
            box += "%5s |       |\n" % ""
            box += "%5.1f |       |%5.1f     <-- Box of mask map\n" % (x1, x2)
            box += "%5s |       |\n" % ""
            box += "%5s ---------\n" % ""
            box += "%5s %5.1f\n" % ("", y2)

            # print box

            print("%2s %-17s %10s %8s" % ("No", "Name", "time[s]", "%"))

            msg = (
                "Coordinates: x = "
                + str(coord[i * 2])
                + "  y = "
                + str(coord[i * 2 + 1])
                + " of gauge is outside mask map\n\n"
            )
            msg += box
            msg += '\nPlease have a look at "MaskMap" or "Gauges"'
            raise Exception(msg)
    if returnmap:
        mapnp = compressArray(null).astype(np.int64)
        return mapnp
    else:
        return col, row


def setmaskmapAttr(x, y, col, row, cell):
    """
    Definition of cell size, coordinates of the meteo maps and maskmap

    :param x: upper left corner x
    :param y: upper left corner y
    :param col: number of cols
    :param row: number of rows
    :param cell: cell size
    :return: -
    """
    invcell = round(1 / cell, 0)
    # getgeotransform only delivers single precision!
    cell = 1 / invcell
    if (x - int(x)) != 0.0:
        if abs(x - int(x)) > 1e9:
            x = 1 / round(1 / (x - int(x)), 4) + int(x)
        else:
            x = round(x, 4)
    if (y - int(y)) != 0.0:
        if abs(y - int(y)) > 1e9:
            y = 1 / round(1 / (y - int(y)), 4) + int(y)
        else:
            y = round(y, 4)

    maskmapAttr["x"] = x
    maskmapAttr["y"] = y
    maskmapAttr["col"] = col
    maskmapAttr["row"] = row
    maskmapAttr["cell"] = cell
    maskmapAttr["invcell"] = invcell


def loadsetclone(self, name):
    """
    load the maskmap and set as clone

    :param name: name of mask map, can be a file or - row col cellsize xupleft yupleft -
    :return: new mask map

    """

    filename = cbinding(name)

    nf2 = gdal.Open(str(filename), gdalconst.GA_ReadOnly)
    geotransform = nf2.GetGeoTransform()
    geotrans.append(geotransform)
    setmaskmapAttr(
        geotransform[0],
        geotransform[3],
        nf2.RasterXSize,
        nf2.RasterYSize,
        geotransform[1],
    )

    band = nf2.GetRasterBand(1)
    # bandtype = gdal.GetDataTypeName(band.DataType)
    mapnp = band.ReadAsArray(0, 0, nf2.RasterXSize, nf2.RasterYSize)
    # 10 because that includes all valid LDD values [1-9]
    mapnp[mapnp > 10] = 0
    mapnp[mapnp < -10] = 0
    mapnp = np.invert(mapnp.astype(bool))

    flagmap = True

    if Flags["check"]:
        checkmap(name, filename, mapnp, flagmap, False, 0)

    maskldd = loadmap("Ldd", compress=False)
    # make sure mapnp dtype is bool
    assert mapnp.dtype == bool
    mask = np.logical_not(np.logical_and(maskldd, mapnp))

    #    mask=np.isnan(mapnp)
    #    mask[mapnp==0] = True # all 0 become mask out
    mapC = np.ma.compressed(np.ma.masked_array(mask, mask))

    # Definition of compressed array and info how to blow it up again
    maskinfo["mask"] = mask
    maskinfo["shape"] = mask.shape
    maskinfo["maskflat"] = mask.ravel()  # map to 1D not compresses
    maskinfo["shapeflat"] = maskinfo["maskflat"].shape  # length of the 1D array
    maskinfo["mapC"] = mapC.shape  # length of the compressed 1D array
    maskinfo["maskall"] = np.ma.masked_all(
        maskinfo["shapeflat"]
    )  # empty map 1D but with mask
    maskinfo["maskall"].mask = maskinfo["maskflat"]

    globals.inZero = np.zeros(maskinfo["mapC"])

    if Flags["check"]:
        checkmap("Mask+Ldd", "", np.ma.masked_array(mask, mask), flagmap, True, mapC)

    # if the final results map should be cover up with some mask:
    if "coverresult" in binding:
        coverresult[0] = returnBool("coverresult")
        if coverresult[0]:
            cover = loadmap("covermap", compress=False)
            cover[cover > 1] = False
            cover[cover == 1] = True
            coverresult[1] = cover

    return mapC


def maskfrompoint(mask2D, xleft, yup):
    """
    load a static map either value or pc raster map or netcdf

    :param mask2D: 2D array of new mask
    :param xleft: left lon coordinate
    :param yup: upper lat coordinate
    :return:  new mask map
    """

    if xleft == -1:
        msg = "MaskMap point does not have a valid value in the river network (LDD)"
        raise Exception(msg)

    x = xleft * maskmapAttr["cell"] + maskmapAttr["x"]
    y = maskmapAttr["y"] - yup * maskmapAttr["cell"]

    maskmapAttr["x"] = x
    maskmapAttr["y"] = y
    maskmapAttr["col"] = mask2D.shape[1]
    maskmapAttr["row"] = mask2D.shape[0]

    mask = np.invert(np.bool8(mask2D))
    mapC = np.ma.compressed(np.ma.masked_array(mask, mask))

    # Definition of compressed array and info how to blow it up again
    maskinfo["mask"] = mask
    maskinfo["shape"] = mask.shape
    maskinfo["maskflat"] = mask.ravel()  # map to 1D not compresses
    maskinfo["shapeflat"] = maskinfo["maskflat"].shape  # length of the 1D array
    maskinfo["mapC"] = mapC.shape  # length of the compressed 1D array
    maskinfo["maskall"] = np.ma.masked_all(
        maskinfo["shapeflat"]
    )  # empty map 1D but with mask
    maskinfo["maskall"].mask = maskinfo["maskflat"]

    globals.inZero = np.zeros(maskinfo["mapC"])
    return mapC


def loadmap(name, compress=True, local=False, cut=True):
    """
    load a static map either value or pc raster map or netcdf

    :param name: name of map
    :param compress: if True the return map will be compressed
    :param local: if True the map is local and will be not cut
    :param cut: if True the map will be not cut
    :return:  1D numpy array of map
    """

    value = str(cbinding(name))
    filename = value

    try:  # loading an integer or float but not a map
        mapC = float(value)
        flagmap = False
        load = True
        if Flags["check"]:
            checkmap(name, filename, mapC, False, False, 0)
        return mapC
    except ValueError:
        load = False

    if os.path.splitext(value)[1].startswith(
        ".nc"
    ):  # read a netcdf  (single one not a stack)
        # get mapextend of netcdf map and calculate the cutting
        # cut0, cut1, cut2, cut3 = mapattrNetCDF(filename)

        filename, varname = value.split(":")

        nf1 = Dataset(filename, "r")
        cut0, cut1, cut2, cut3 = mapattrNetCDF(filename, check=False)

        # load netcdf map but only the rectangle needed
        # nf1 = Dataset(filename, 'r')

        if not timestepInit:
            # with np.errstate(invalid='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # in order to ignore some invalid value comments
                if cut:
                    mapnp = nf1.variables[varname][cut2:cut3, cut0:cut1].astype(
                        np.float64
                    )
                else:
                    mapnp = nf1.variables[varname][:]
        else:
            if "time" in nf1.variables:
                timestepI = Calendar(timestepInit[0])
                if type(timestepI) is datetime.datetime:
                    timestepI = date2num(timestepI, nf1.variables["time"].units)
                else:
                    timestepI = int(timestepI) - 1

                if not (timestepI in nf1.variables["time"][:]):
                    msg = (
                        "time step "
                        + str(int(timestepI) + 1)
                        + " not stored in "
                        + filename
                    )
                    raise Exception(msg)
                itime = np.where(nf1.variables["time"][:] == timestepI)[0][0]
                if cut:
                    mapnp = nf1.variables[varname][itime, cut2:cut3, cut0:cut1]
                else:
                    mapnp = nf1.variables[varname][itime][:]
            else:
                if cut:
                    mapnp = nf1.variables[varname][cut2:cut3, cut0:cut1]
                else:
                    mapnp = nf1.variables[varname][:]

        nf1.close()

    else:

        filename = str(cbinding(name))
        try:
            nf2 = gdal.Open(filename, gdalconst.GA_ReadOnly)
            band = nf2.GetRasterBand(1)
            mapnp = band.ReadAsArray(0, 0, nf2.RasterXSize, nf2.RasterYSize).astype(
                np.float64
            )
            # if local no cut
            if not local:
                if cut:
                    cut0, cut1, cut2, cut3 = mapattrTiff(nf2)
                    mapnp = mapnp[cut2:cut3, cut0:cut1]
        except:
            raise

    if compress:
        mapC = compressArray(mapnp, name=filename)
        if Flags["check"]:
            checkmap(name, filename, mapnp, True, True, mapC)
    else:
        mapC = mapnp
        if Flags["check"]:
            checkmap(name, filename, mapnp, True, False, 0)

    mapC = reduce_precision(mapC)
    return mapC


# -----------------------------------------------------------------------
# Compressing to 1-dimensional numpy array
# -----------------------------------------------------------------------


def compressArray(map, name="None", zeros=0.0):
    """
    Compress 2D array with missing values to 1D array without missing values

    :param map: in map
    :param name: filename of the map
    :param zeros: add zeros (default= 0) if values of map are to big or too small
    :return: Compressed 1D array
    """

    mapnp1 = np.ma.masked_array(map, maskinfo["mask"])
    mapC = np.ma.compressed(mapnp1)
    # if fill: mapC[np.isnan(mapC)]=0
    if name != "None":
        if np.max(np.isnan(mapC)):
            msg = name + " has less valid pixels than area or ldd \n"
            raise Exception(msg)
            # test if map has less valid pixel than area.map (or ldd)
    # if a value is bigger or smaller than 1e20, -1e20 than the standard value is taken
    mapC[mapC > 1.0e20] = zeros
    mapC[mapC < -1.0e20] = zeros

    return mapC


def decompress(map):
    """
    Decompress 1D array without missing values to 2D array with missing values

    :param map: numpy 1D array as input
    :return: 2D array for displaying
    """

    # dmap=np.ma.masked_all(maskinfo['shapeflat'], dtype=map.dtype)
    dmap = maskinfo["maskall"].copy()
    dmap[~maskinfo["maskflat"]] = map[:]
    dmap = dmap.reshape(maskinfo["shape"])

    # check if integer map (like outlets, lakes etc
    try:
        checkint = str(map.dtype)
    except:
        checkint = "x"
    if checkint == "int16" or checkint == "int32":
        dmap[dmap.mask] = -9999
    elif checkint == "int8":
        dmap[dmap < 0] = 0
    else:
        dmap[dmap.mask] = -9999

    return dmap


# -----------------------------------------------------------------------
# NETCDF
# -----------------------------------------------------------------------


def getmeta(key, varname, alternative):
    """
    get the meta data information for the netcdf output from the global
    variable metaNetcdfVar

    :param key: key
    :param varname: variable name e.g. self.var.Precipitation
    :return: metadata information
    """

    ret = alternative
    if varname in metaNetcdfVar:
        if key in metaNetcdfVar[varname]:
            ret = metaNetcdfVar[varname][key]
    return ret


def metaNetCDF():
    """
    get the map metadata from precipitation netcdf maps
    """

    try:
        name = cbinding("PrecipitationMaps")
        name1 = glob.glob(os.path.normpath(name))[0]
        nf1 = Dataset(name1, "r")
        for var in nf1.variables:
            metadataNCDF[var] = nf1.variables[var].__dict__
        nf1.close()
    except:
        msg = "Trying to get metadata from netcdf \n"
        raise


def readCoord(name):
    """
    get the meta data information for the netcdf output from the global
    variable metaNetcdfVar
    :param name: name of the netcdf file
    :return: latitude, longitude, cell size, inverse cell size
    """

    namenc = os.path.splitext(name)[0] + ".nc"

    try:
        nf1 = Dataset(namenc, "r")
        nc = True
    except:
        nc = False
    if nc:
        lat, lon, cell, invcell, rows, cols = readCoordNetCDF(namenc)
    else:
        raster = gdal.Open(name)
        rows = raster.RasterYSize
        cols = raster.RasterXSize
        gt = raster.GetGeoTransform()

        cell = gt[1]
        invcell = round(1.0 / cell, 0)
        if invcell == 0:
            invcell = 1.0 / cell

        # getgeotransform only delivers single precision!
        cell = 1 / invcell
        lon = gt[0]
        lat = gt[3]
        # lon = 1 / round(1 / (x1 - int(x1)), 4) + int(x1)
        # lat = 1 / round(1 / (y1 - int(y1)), 4) + int(y1)

    return lat, lon, cell, invcell, rows, cols


def readCoordNetCDF2(name, check=True):

    if check:
        try:
            nf1 = Dataset(name, "r")
        except:
            msg = "Checking netcdf map \n"
            raise
    else:
        # if subroutine is called already from inside a try command
        nf1 = Dataset(name, "r")

    lons = nf1.variables["lon"][:]
    lats = nf1.variables["lat"][:]
    nf1.close()
    return lats, lons


def readCoordNetCDF(name, check=True):
    """
    reads the map attributes col, row etc from a netcdf map
    :param name: name of the netcdf file
    :param check:  checking if netcdffile exists
    :return: latitude, longitude, cell size, inverse cell size
    """

    if check:
        try:
            nf1 = Dataset(name, "r")
        except:
            msg = "Error 205: Checking netcdf map \n"
            raise
    else:
        # if subroutine is called already from inside a try command
        nf1 = Dataset(name, "r")

    if not ("coordx" in maskmapAttr.keys()):
        if "lon" in nf1.variables.keys():
            maskmapAttr["coordx"] = "lon"
            maskmapAttr["coordy"] = "lat"
        else:
            maskmapAttr["coordx"] = "x"
            maskmapAttr["coordy"] = "y"

    rows = nf1.variables[maskmapAttr["coordy"]].shape[0]
    cols = nf1.variables[maskmapAttr["coordx"]].shape[0]

    lon0 = nf1.variables[maskmapAttr["coordx"]][0]
    lon1 = nf1.variables[maskmapAttr["coordx"]][1]
    lat0 = nf1.variables[maskmapAttr["coordy"]][0]
    latlast = nf1.variables[maskmapAttr["coordy"]][-1]
    nf1.close()
    # swap to make lat0 the biggest number
    if lat0 < latlast:
        lat0, latlast = latlast, lat0

    cell = round(np.abs(lon1 - lon0), 8)
    invcell = round(1.0 / cell, 0)
    if invcell == 0:
        invcell = 1.0 / cell

    lon = round(lon0 - cell / 2, 8)
    lat = round(lat0 + cell / 2, 8)

    return lat, lon, cell, invcell, rows, cols


def readCalendar(name):
    nf1 = Dataset(name, "r")
    dateVar["calendar"] = nf1.variables["time"].calendar
    nf1.close()


def checkMeteo_Wordclim(meteodata, wordclimdata):
    """
    reads the map attributes of meteo dataset and wordclima dataset
    and compare if it has the same map extend

    :param nmeteodata: name of the meteo netcdf file
    :param wordlclimdata:  cname of the wordlclim netcdf file
    :return: True if meteo and wordclim has the same mapextend
    """

    try:
        nf1 = Dataset(meteodata.split(":")[0], "r")
    except:
        msg = "Checking netcdf map \n"
        raise

    lonM0 = nf1.variables["lon"][0]
    lon1 = nf1.variables["lon"][1]
    cellM = round(np.abs(lon1 - lonM0) / 2.0, 8)
    lonM0 = round(lonM0 - cellM, 8)

    lonM1 = round(nf1.variables["lon"][-1] + cellM, 8)
    latM0 = nf1.variables["lat"][0]
    latM1 = nf1.variables["lat"][-1]
    nf1.close()

    # swap to make lat0 the biggest number
    if latM0 < latM1:
        latM0, latM1 = latM1, latM0
    latM0 = round(latM0 + cellM, 8)
    latM1 = round(latM1 - cellM, 8)

    # load Wordclima data
    try:
        nf1 = Dataset(wordclimdata.split(":")[0], "r")
    except:
        msg = "Checking netcdf map \n"
        raise

    lonW0 = nf1.variables["lon"][0]
    lon1 = nf1.variables["lon"][1]
    cellW = round(np.abs(lon1 - lonW0) / 2.0, 8)
    lonW0 = round(lonW0 - cellW, 8)
    lonW1 = round(nf1.variables["lon"][-1] + cellW, 8)

    latW0 = nf1.variables["lat"][0]
    latW1 = nf1.variables["lat"][-1]
    nf1.close()
    # swap to make lat0 the biggest number
    if latW0 < latW1:
        latW0, latW1 = latW1, latW0
    latW0 = round(latW0 + cellW, 8)
    latW1 = round(latW1 - cellW, 8)
    # calculate the controll variable
    contr1 = lonM0 + lonM1 + latM0 + latM1
    contr2 = lonW0 + lonW1 + latW0 + latW1
    contr = abs(round(contr1 - contr2, 5))

    check = True
    if contr > 0.00001:
        # msg = "Data from meteo dataset and Wordclim dataset does not match"
        # raise Exception(msg)
        check = False

    return check


def mapattrNetCDF(name, check=True):
    """
    get the 4 corners of a netcdf map to cut the map
    defines the rectangular of the mask map inside the netcdf map
    calls function :meth:`management_modules.data_handling.readCoord`
    :param name: name of the netcdf file
    :param check:  checking if netcdffile exists
    :return: cut1,cut2,cut3,cut4
    :raises if cell size is different: :meth:`management_modules.messages.Exception`
    """

    lat, lon, cell, invcell, rows, cols = readCoord(name)

    if maskmapAttr["invcell"] != invcell:
        msg = (
            "Error 107: Cell size different in maskmap: "
            + binding["MaskMap"]
            + " and: "
            + name
        )
        raise Exception(msg)

    xx = maskmapAttr["x"]
    yy = maskmapAttr["y"]
    # cut0 = int(0.0001 + np.abs(xx - lon) * invcell) # argmin() ??
    # cut2 = int(0.0001 + np.abs(yy - lat) * invcell) #
    cut0 = int(np.abs(xx + maskmapAttr["cell"] / 2 - lon) * invcell)  # argmin() ??
    cut2 = int(np.abs(yy - maskmapAttr["cell"] / 2 - lat) * invcell)  #

    cut1 = cut0 + maskmapAttr["col"]
    cut3 = cut2 + maskmapAttr["row"]
    return cut0, cut1, cut2, cut3


def mapattrNetCDFMeteo(name, check=True):
    """
    get the map attributes like col, row etc from a netcdf map
    and define the rectangular of the mask map inside the netcdf map
    calls function :meth:`management_modules.data_handling.readCoordNetCDF`
    :param name: name of the netcdf file
    :param check:  checking if netcdffile exists
    :return: cut0,cut1,cut2,cut3,cut4,cut5,cut6,cut7
    """

    lat, lon, cell, invcell, rows, cols = readCoordNetCDF(name, check)

    # x0,xend, y0,yend - borders of fine resolution map
    lon0 = maskmapAttr["x"]
    lat0 = maskmapAttr["y"]
    lonend = lon0 + maskmapAttr["col"] / maskmapAttr["invcell"]
    latend = lat0 - maskmapAttr["row"] / maskmapAttr["invcell"]

    # cut for 0.5 deg map based on finer resolution

    # lats = nc_simulated.variables['lat'][:]
    # in_lat = discharge_location[1]
    # lat_idx = geo_idx(in_lat, lats)
    # geo_idx = (np.abs(dd_array - dd)).argmin()
    # geo_idx(dd, dd_array):

    cut0 = math.floor(0.0001 + np.abs(lon0 - lon) * invcell)
    cut2 = math.floor(0.0001 + np.abs(lat0 - lat) * invcell)

    # lon and lat of coarse meteo dataset
    lonCoarse = (cut0 * cell) + lon
    latCoarse = lat - (cut2 * cell)
    cut4 = int(0.0001 + np.abs(lon0 - lonCoarse) * maskmapAttr["invcell"])
    cut5 = cut4 + maskmapAttr["col"]
    cut6 = int(0.0001 + np.abs(lat0 - latCoarse) * maskmapAttr["invcell"])
    cut7 = cut6 + maskmapAttr["row"]

    # now coarser cut of the coarse meteo dataset
    cut1 = math.ceil(0.0001 + np.abs(lonend - lon) * invcell)
    cut3 = math.ceil(0.0001 + np.abs(latend - lat) * invcell)

    # test if fine cut is inside coarse cut
    cellx = (cut1 - cut0) * maskmapAttr["reso_mask_meteo"]
    celly = (cut3 - cut2) * maskmapAttr["reso_mask_meteo"]

    if cellx < cut5:
        cut1 += 1
    if celly < cut7:
        cut3 += 1

    if maskmapAttr["coordy"] == "lat":
        if cut1 > (360 * invcell):
            cut1 = int(360 * invcell)
        if cut3 > (180 * invcell):
            cut3 = int(180 * invcell)

    return cut0, cut1, cut2, cut3, cut4, cut5, cut6, cut7


def mapattrTiff(nf2):
    """
    map attributes of a geotiff file

    :param nf2:
    :return: cut0,cut1,cut2,cut3
    """

    geotransform = nf2.GetGeoTransform()
    x1 = geotransform[0]
    y1 = geotransform[3]

    # maskmapAttr['col'] = nf2.RasterXSize
    # maskmapAttr['row'] = nf2.RasterYSize
    cellSize = geotransform[1]

    invcell = round(1 / cellSize, 0)

    # getgeotransform only delivers single precision!
    cellSize = 1 / invcell
    x1 = 1 / round(1 / (x1 - int(x1)), 4) + int(x1)
    y1 = 1 / round(1 / (y1 - int(y1)), 4) + int(y1)

    if maskmapAttr["invcell"] != invcell:
        msg = "Cell size different in maskmap: " + binding["MaskMap"]
        raise Exception(msg)

    x = x1 - cellSize / 2
    y = y1 + cellSize / 2
    cut0 = int(0.01 + np.abs(maskmapAttr["x"] - x) * invcell)
    cut2 = int(0.01 + np.abs(maskmapAttr["y"] - y) * invcell)

    cut1 = cut0 + maskmapAttr["col"]
    cut3 = cut2 + maskmapAttr["row"]

    return cut0, cut1, cut2, cut3


def writenetcdf(
    var,
    netfile,
    prename,
    addname,
    varunits,
    inputmap,
    timeStamp,
    posCnt,
    flag,
    flagTime,
    nrdays=None,
    dateunit="days",
):
    """
    write a netcdf stack

    :param netfile: file name
    :param prename: 1st part of variable name with tell which variable e.g. discharge
    :param addname: part of the variable name with tells about the timestep e.g. daily, monthly
    :param varunits: unit of the variable
    :param inputmap: 1D array to be put as netcdf
    :param timeStamp: time
    :param posCnt: calculate nummer of the indece for time
    :param flag: to indicate if the file is new -> netcdf header has to be written,or simply appending data
    :param flagtime: to indicate the variable is time dependend (not a single array!)
    :param nrdays: (optional) if indicate number of days are set in the time variable (makes files smaller!)
    :param dateunit: (optional) dateunit indicate if the timestep in netcdf is days, month or years
    :return: flag: to indicate if the file is set up
    """

    # check if it is a modflow grid which has another resolution
    if "modflow" in prename.lower():
        modflow = True
        row = domain["nrow"]
        col = domain["ncol"]
        metadataNCDF["modflow_x"] = {}
        metadataNCDF["modflow_x"]["standard_name"] = "UTM_X"
        metadataNCDF["modflow_x"]["units"] = "m"
        metadataNCDF["modflow_y"] = {}
        metadataNCDF["modflow_y"]["standard_name"] = "UTM_Y"
        metadataNCDF["modflow_y"]["units"] = "m"
    else:
        modflow = False
        row = np.abs(cutmap[3] - cutmap[2]) * var.scaling
        col = np.abs(cutmap[1] - cutmap[0]) * var.scaling

    # create real varname with variable name + time depending name e.g. discharge + monthavg
    varname = prename + addname

    if not flag:
        nf1 = Dataset(netfile, "w", format="NETCDF4")

        # general Attributes
        settings = os.path.realpath(settingsfile[0])
        nf1.settingsfile = settings + ": " + xtime.ctime(os.path.getmtime(settings))
        nf1.run_created = xtime.ctime(xtime.time())
        nf1.Source_Software = "CWATM Python: " + versioning["exe"]
        nf1.Platform = versioning["platform"]
        nf1.Version = (
            versioning["version"]
            + ": "
            + versioning["lastfile"]
            + " "
            + versioning["lastdate"]
        )
        nf1.institution = cbinding("institution")
        nf1.title = cbinding("title")
        nf1.source = "CWATM output maps"
        nf1.Conventions = "CF-1.6"

        # put the additional genaral meta data information from the xml file into the netcdf file
        # infomation from the settingsfile comes first
        if prename in metaNetcdfVar:
            for key in metaNetcdfVar[prename]:
                if not (key in list(nf1.__dict__.keys())):
                    if not (key in ["unit", "long_name", "standard_name"]):
                        nf1.__setattr__(key, metaNetcdfVar[prename][key])

        # Dimension
        if modflow:
            lon = nf1.createDimension("x", col)  # x 1000
            longitude = nf1.createVariable("x", "f8", ("x",))
            for i in metadataNCDF["modflow_x"]:
                exec('%s="%s"' % ("longitude." + i, metadataNCDF["modflow_x"][i]))
            lat = nf1.createDimension("y", row)  # x 950
            latitude = nf1.createVariable("y", "f8", "y")
            for i in metadataNCDF["modflow_y"]:
                exec('%s="%s"' % ("latitude." + i, metadataNCDF["modflow_y"][i]))

        else:
            if "x" in list(metadataNCDF.keys()):
                lon = nf1.createDimension("x", col)  # x 1000
                longitude = nf1.createVariable("x", "f8", ("x",))
                for i in metadataNCDF["x"]:
                    exec('%s="%s"' % ("longitude." + i, metadataNCDF["x"][i]))
            if "lon" in list(metadataNCDF.keys()):
                lon = nf1.createDimension("lon", col)
                longitude = nf1.createVariable("lon", "f8", ("lon",))
                for i in metadataNCDF["lon"]:
                    exec('%s="%s"' % ("longitude." + i, metadataNCDF["lon"][i]))
            if "y" in list(metadataNCDF.keys()):
                lat = nf1.createDimension("y", row)  # x 950
                latitude = nf1.createVariable("y", "f8", "y")
                for i in metadataNCDF["y"]:
                    exec('%s="%s"' % ("latitude." + i, metadataNCDF["y"][i]))
            if "lat" in list(metadataNCDF.keys()):
                lat = nf1.createDimension("lat", row)  # x 950
                latitude = nf1.createVariable("lat", "f8", "lat")
                for i in metadataNCDF["lat"]:
                    exec('%s="%s"' % ("latitude." + i, metadataNCDF["lat"][i]))

        # projection
        if "laea" in list(metadataNCDF.keys()):
            proj = nf1.createVariable("laea", "i4")
            for i in metadataNCDF["laea"]:
                exec('%s="%s"' % ("proj." + i, metadataNCDF["laea"][i]))
        if "lambert_azimuthal_equal_area" in list(metadataNCDF.keys()):
            proj = nf1.createVariable("lambert_azimuthal_equal_area", "i4")
            for i in metadataNCDF["lambert_azimuthal_equal_area"]:
                exec(
                    '%s="%s"'
                    % ("proj." + i, metadataNCDF["lambert_azimuthal_equal_area"][i])
                )

        # Fill variables
        if modflow:
            lats = np.arange(
                domain["north"], domain["south"] - 1, domain["rowsize"] * -1
            )
            lons = np.arange(domain["west"], domain["east"] + 1, domain["colsize"])
            # lons =  np.linspace(domain['north'] , domain['south'], col, endpoint=False)
            latitude[:] = lats
            longitude[:] = lons

        else:
            cell = var.cell_size
            xl = maskmapAttr["x"]
            xr = xl + col * cell
            yu = maskmapAttr["y"]
            yd = yu - row * cell
            lats = np.linspace(yu, yd, row, endpoint=False)
            lons = np.linspace(xl, xr, col, endpoint=False)

            latitude[:] = lats - cell / 2.0
            longitude[:] = lons + cell / 2.0

        if flagTime:

            year = dateVar["dateStart"].year
            if year > 1900:
                yearstr = "1901"
            elif year < 1861:
                yearstr = "1650"
            else:
                yearstr = "1861"

            # nf1.createDimension('time', None)
            nf1.createDimension("time", nrdays)
            time = nf1.createVariable("time", "f8", "time")
            time.standard_name = "time"
            if dateunit == "days":
                time.units = "Days since " + yearstr + "-01-01"
            if dateunit == "months":
                time.units = "Months since " + yearstr + "-01-01"
            if dateunit == "years":
                time.units = "Years since " + yearstr + "-01-01"
            # time.calendar = 'standard'
            time.calendar = dateVar["calendar"]

            if modflow:
                value = nf1.createVariable(
                    varname, "f4", ("time", "y", "x"), zlib=True, fill_value=1e20
                )
            else:
                if "x" in list(metadataNCDF.keys()):
                    value = nf1.createVariable(
                        varname, "f4", ("time", "y", "x"), zlib=True, fill_value=1e20
                    )
                if "lon" in list(metadataNCDF.keys()):
                    # value = nf1.createVariable(varname, 'f4', ('time', 'lat', 'lon'), zlib=True, fill_value=1e20)
                    value = nf1.createVariable(
                        varname,
                        "f4",
                        ("time", "lat", "lon"),
                        zlib=True,
                        fill_value=1e20,
                        chunksizes=(1, row, col),
                    )
        else:
            if modflow:
                value = nf1.createVariable(
                    varname, "f4", ("y", "x"), zlib=True, fill_value=1e20
                )
            else:
                if "x" in list(metadataNCDF.keys()):
                    value = nf1.createVariable(
                        varname, "f4", ("y", "x"), zlib=True, fill_value=1e20
                    )
                if "lon" in list(metadataNCDF.keys()):
                    # for world lat/lon coordinates
                    value = nf1.createVariable(
                        varname, "f4", ("lat", "lon"), zlib=True, fill_value=1e20
                    )

        value.standard_name = getmeta("standard_name", prename, varname)
        p1 = getmeta("long_name", prename, prename)
        p2 = getmeta("time", addname, addname)
        value.long_name = p1 + p2
        value.units = getmeta("unit", prename, varunits)

        for key in list(metadataNCDF.keys()):
            if "esri_pe_string" in list(metadataNCDF[key].keys()):
                value.esri_pe_string = metadataNCDF[key]["esri_pe_string"]

    else:
        nf1 = Dataset(netfile, "a")

    if flagTime:
        date_time = nf1.variables["time"]
        if dateunit == "days":
            nf1.variables["time"][posCnt - 1] = date2num(
                timeStamp, date_time.units, date_time.calendar
            )
        if dateunit == "months":
            nf1.variables["time"][posCnt - 1] = (
                (timeStamp.year - 1901) * 12 + timeStamp.month - 1
            )
        if dateunit == "years":
            nf1.variables["time"][posCnt - 1] = timeStamp.year - 1901

        # nf1.variables['time'][posCnt - 1] = 60 + posCnt

    # if inputmap is not an array give out errormessage
    if not (hasattr(inputmap, "__len__")):
        date1 = "%02d/%02d/%02d" % (timeStamp.day, timeStamp.month, timeStamp.year)
        msg = (
            "No values in: "
            + varname
            + " on date: "
            + date1
            + "\nCould not write: "
            + netfile
        )
        nf1.close()
        raise Exception(msg)

    if modflow:
        mapnp = inputmap
    else:
        mapnp = var.decompress(inputmap)
        if isinstance(mapnp, cp.ndarray):
            mapnp = mapnp.get()
        # mapnp[~maskinfo['maskflat']] = inputmap[:]
        # #mapnp = mapnp.reshape(maskinfo['shape']).data
        # mapnp = mapnp.reshape(maskinfo['shape'])

        if coverresult[0]:
            mapnp = mapnp.reshape(maskinfo["shape"]).data
            mapnp = np.where(coverresult[1], mapnp, np.nan)
    if flagTime:
        nf1.variables[varname][posCnt - 1, :, :] = mapnp
    else:
        # without timeflag
        nf1.variables[varname][:, :] = mapnp

    nf1.close()
    flag = True

    return flag


def writeIniNetcdf(netfile, varlist, inputlist):
    """
    write variables to netcdf init file

    :param netfile: file name
    :param varlist: list of variable to be written in the netcdf file
    :param inputlist: stack of 1D arrays
    :return: -
    """

    row = np.abs(cutmap[3] - cutmap[2])
    col = np.abs(cutmap[1] - cutmap[0])

    nf1 = Dataset(netfile, "w", format="NETCDF4")

    # general Attributes
    nf1.settingsfile = os.path.realpath(settingsfile[0])
    nf1.date_created = xtime.ctime(xtime.time())
    nf1.Source_Software = "CWATM Python"
    nf1.institution = cbinding("institution")
    nf1.title = cbinding("title")
    nf1.source = "CWATM initial conditions maps"
    nf1.Conventions = "CF-1.6"

    # put the additional genaral meta data information from the xml file into the netcdf file
    # infomation from the settingsfile comes first

    if "initcondition" in metaNetcdfVar:
        for key in metaNetcdfVar["initcondition"]:
            if not (key in list(nf1.__dict__.keys())):
                if not (key in ["unit", "long_name", "standard_name"]):
                    nf1.__setattr__(key, metaNetcdfVar["initcondition"][key])

    # Dimension
    if "x" in list(metadataNCDF.keys()):
        lon = nf1.createDimension("x", col)  # x 1000
        longitude = nf1.createVariable("x", "f8", ("x",))
        for i in metadataNCDF["x"]:
            exec('%s="%s"' % ("longitude." + i, metadataNCDF["x"][i]))
    if "lon" in list(metadataNCDF.keys()):
        lon = nf1.createDimension("lon", col)
        longitude = nf1.createVariable("lon", "f8", ("lon",))
        for i in metadataNCDF["lon"]:
            exec('%s="%s"' % ("longitude." + i, metadataNCDF["lon"][i]))
    if "y" in list(metadataNCDF.keys()):
        lat = nf1.createDimension("y", row)  # x 950
        latitude = nf1.createVariable("y", "f8", "y")
        for i in metadataNCDF["y"]:
            exec('%s="%s"' % ("latitude." + i, metadataNCDF["y"][i]))
    if "lat" in list(metadataNCDF.keys()):
        lat = nf1.createDimension("lat", row)  # x 950
        latitude = nf1.createVariable("lat", "f8", "lat")
        for i in metadataNCDF["lat"]:
            exec('%s="%s"' % ("latitude." + i, metadataNCDF["lat"][i]))

    # projection
    if "laea" in list(metadataNCDF.keys()):
        proj = nf1.createVariable("laea", "i4")
        for i in metadataNCDF["laea"]:
            exec('%s="%s"' % ("proj." + i, metadataNCDF["laea"][i]))
    if "lambert_azimuthal_equal_area" in list(metadataNCDF.keys()):
        proj = nf1.createVariable("lambert_azimuthal_equal_area", "i4")
        for i in metadataNCDF["lambert_azimuthal_equal_area"]:
            exec(
                '%s="%s"'
                % ("proj." + i, metadataNCDF["lambert_azimuthal_equal_area"][i])
            )

    # Fill variables
    cell = maskmapAttr["cell"]
    xl = maskmapAttr["x"]
    xr = xl + col * cell
    yu = maskmapAttr["y"]
    yd = yu - row * cell
    lats = np.linspace(yu, yd, row, endpoint=False)
    lons = np.linspace(xl, xr, col, endpoint=False)

    latitude[:] = lats - cell / 2.0
    longitude[:] = lons + cell / 2.0

    i = 0
    for varname in varlist:

        if "x" in list(metadataNCDF.keys()):
            value = nf1.createVariable(
                varname, "f8", ("y", "x"), zlib=True, fill_value=1e20
            )
        if "lon" in list(metadataNCDF.keys()):
            # for world lat/lon coordinates
            value = nf1.createVariable(
                varname, "f8", ("lat", "lon"), zlib=True, fill_value=1e20
            )

        value.standard_name = getmeta("standard_name", varname, varname)
        value.long_name = getmeta("long_name", varname, varname)
        value.units = getmeta("unit", varname, "undefined")

        # write values
        mapnp = maskinfo["maskall"].copy()
        help = np.minimum(10e15, np.maximum(-9999.0, inputlist[i][:]))
        mapnp[~maskinfo["maskflat"]] = help
        # mapnp = mapnp.reshape(maskinfo['shape']).data
        mapnp = mapnp.reshape(maskinfo["shape"])

        nf1.variables[varname][:, :] = mapnp
        i += 1

    nf1.close()


# --------------------------------------------------------------------------------------------
# report .tif and .maps


def report(name, valueIn, compr=True):
    """
    For debugging: Save the 2D array as .map or .tif

    :param name: Filename of the map
    :param valueIn: 1D or 2D array in
    :param compr: (optional) array is 1D (default) or 2D
    :return: -

    ::

        Example:
        > report(c:/temp/ksat1.map, self_.var_.ksat1)

    """
    filename = os.path.splitext(name)
    pcmap = False
    if filename[1] == ".map":
        pcmap = True

    if compr:
        value = decompress(valueIn)
    else:
        value = valueIn

    checkint = value.dtype.char in np.typecodes["AllInteger"]
    ny, nx = value.shape

    if pcmap:  # if it is a map
        raster = gdal.GetDriverByName("PCRaster")
        # ds = raster.Create(name, nx, ny, 1, gdal.GDT_Float32)
        if checkint:
            ds = raster.Create(
                name, nx, ny, 1, gdal.GDT_Int32, ["PCRASTER_VALUESCALE=VS_NOMINAL"]
            )
        else:
            ds = raster.Create(
                name, nx, ny, 1, gdal.GDT_Float32, ["PCRASTER_VALUESCALE=VS_SCALAR"]
            )

        ds.SetGeoTransform(geotrans[0])  # specify coords
        outband = ds.GetRasterBand(1)
        # set NoData value
        # outband.SetNoDataValue(np.nan)
        outband.SetNoDataValue(-9999)
        value[np.isnan(value)] = -9999

    else:  # if is not a .map
        if checkint:
            ds = gdal.GetDriverByName("GTiff").Create(
                name, nx, ny, 1, gdal.GDT_Int32, ["COMPRESS=LZW"]
            )
        else:
            ds = gdal.GetDriverByName("GTiff").Create(
                name, nx, ny, 1, gdal.GDT_Float32, ["COMPRESS=LZW"]
            )

        ds.SetGeoTransform(geotrans[0])  # specify coords
        srs = osr.SpatialReference()  # establish encoding
        srs.ImportFromEPSG(4326)  # WGS84 lat/long
        ds.SetProjection(srs.ExportToWkt())  # export coords to file
        outband = ds.GetRasterBand(1)
        # set NoData value
        outband.SetNoDataValue(-9999)
        outband.SetStatistics(
            np.nanmin(value).astype(np.float),
            np.nanmax(value).astype(np.float),
            np.nanmean(value).astype(np.float),
            np.nanstd(value).astype(np.float),
        )

    outband.WriteArray(value)
    ds.FlushCache()
    ds = None
    outband = None


def returnBool(inBinding):
    """
    Test if parameter is a boolean and return an error message if not, and the boolean if everything is ok

    :param inBinding: parameter in settings file
    :return: boolean of inBinding
    """

    b = cbinding(inBinding)
    assert isinstance(b, bool)
    return b


def checkOptionOptinal(inBinding):
    """
    Test if parameter is a boolean and return an error message if not, and the boolean if everything is ok

    :param inBinding: parameter in settings file
    :return: boolean of inBinding
    """

    b = cbinding(inBinding)
    btrue = b.lower() in ("yes", "true", "t", "1")
    bfalse = b.lower() in ("no", "false", "f", "0")
    if btrue or bfalse:
        return btrue
    else:
        msg = 'Value in: "' + inBinding + '" is not True or False! \nbut: ' + b
        raise Exception(msg)


def checkOption(inBinding):
    """
    Check if option in settings file has a counterpart in the source code

    :param inBinding: parameter in settings file
    """
    return option[inBinding]


def cbinding(inBinding):
    """
    Check if variable in settings file has a counterpart in the source code

    :param inBinding: parameter in settings file
    """
    return binding[inBinding]


def divideValues(x, y, default=0.0):
    """
    returns the result of a division that possibly involves a zero

    :param x:
    :param y: divisor
    :param default: return value if y =0
    :return: result of :math:`x/y` or default if y = 0
    """
    # y1 = y.copy()
    # y1[y1 == 0.] = 1.0
    # z = x / y1
    # z[y == 0.] = default

    # with np.errstate(invalid='ignore', divide='ignore'):
    #    z = np.where(y > 0., x/y, default)
    # have to solve this without err handler to get the error message back
    return np.nan_to_num(x / y)


@njit(cache=True)
def downscale_volume(
    data_gt: Tuple[float, float, float, float, float, float],
    model_gt: Tuple[float, float, float, float, float, float],
    data: np.ndarray,
    mask: np.ndarray,
    grid_to_HRU_uncompressed: np.ndarray,
    downscale_mask: np.ndarray,
    HRU_land_size: np.ndarray,
) -> np.ndarray:

    xoffset = (model_gt[0] - data_gt[0]) / model_gt[1]
    assert 0.0001 > xoffset - round(xoffset) > -0.0001
    xoffset = round(xoffset)
    assert xoffset >= 0

    yoffset = (model_gt[3] - data_gt[3]) / model_gt[5]
    assert 0.0001 > yoffset - round(yoffset) > -0.0001
    yoffset = round(yoffset)
    assert yoffset >= 0

    xratio = data_gt[1] / model_gt[1]
    assert 0.0001 > xratio - round(xratio) > -0.0001
    assert xratio > 0
    xratio = round(xratio)

    yratio = data_gt[5] / model_gt[5]
    assert 0.0001 > yratio - round(yratio) > -0.0001
    assert yratio > 0
    yratio = round(yratio)

    downscale_invmask = ~downscale_mask
    assert xratio > 0
    assert yratio > 0
    assert xoffset > 0
    assert yoffset > 0
    ysize, xsize = data.shape
    yvarsize, xvarsize = mask.shape
    downscaled_array = np.zeros(HRU_land_size.size, dtype=np.float32)
    i = 0
    for y in range(ysize):
        y_left = y * yratio - yoffset
        y_right = min(y_left + yratio, yvarsize)
        y_left = max(y_left, 0)
        for x in range(xsize):
            x_left = x * xratio - xoffset
            x_right = min(x_left + xratio, xvarsize)
            x_left = max(x_left, 0)

            land_area_cell = 0
            for yvar in range(y_left, y_right):
                for xvar in range(x_left, x_right):
                    if mask[yvar, xvar] == False:
                        k = yvar * xvarsize + xvar
                        HRU_right = grid_to_HRU_uncompressed[k]
                        # assert HRU_right != -1
                        if k > 0:
                            HRU_left = grid_to_HRU_uncompressed[k - 1]
                            # assert HRU_left != -1
                        else:
                            HRU_left = 0
                        land_area_cell += (
                            downscale_invmask[HRU_left:HRU_right]
                            * HRU_land_size[HRU_left:HRU_right]
                        ).sum()
                        i += 1

            if land_area_cell:
                for yvar in range(y_left, y_right):
                    for xvar in range(x_left, x_right):
                        if mask[yvar, xvar] == False:
                            k = yvar * xvarsize + xvar
                            HRU_right = grid_to_HRU_uncompressed[k]
                            # assert HRU_right != -1
                            if k > 0:
                                HRU_left = grid_to_HRU_uncompressed[k - 1]
                                # assert HRU_left != -1
                            else:
                                HRU_left = 0
                            downscaled_array[HRU_left:HRU_right] = (
                                downscale_invmask[HRU_left:HRU_right]
                                * HRU_land_size[HRU_left:HRU_right]
                                / land_area_cell
                                * data[y, x]
                            )

    assert i == mask.size - mask.sum()
    return downscaled_array
