# -------------------------------------------------------------------------
# Name:        Data handling
# Purpose:     Transforming netcdf to numpy arrays, checking mask file
#
# Author:      PB
#
# Created:     13/07/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------

import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np

from cwatm.globals import (
    binding,
    maskinfo,
    option,
)

from osgeo import gdal, gdalconst


def loadsetclone(name):
    """
    load the maskmap and set as clone

    :param name: name of mask map, can be a file or - row col cellsize xupleft yupleft -
    :return: new mask map

    """

    filename = cbinding(name)

    nf2 = gdal.Open(str(filename), gdalconst.GA_ReadOnly)

    band = nf2.GetRasterBand(1)
    # bandtype = gdal.GetDataTypeName(band.DataType)
    mapnp = band.ReadAsArray(0, 0, nf2.RasterXSize, nf2.RasterYSize)
    # 10 because that includes all valid LDD values [1-9]
    mapnp[mapnp > 10] = 0
    mapnp[mapnp < -10] = 0
    mapnp = np.invert(mapnp.astype(bool))

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

    return mapC


def loadmap(name, compress=True):
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
        return mapC
    except ValueError:
        pass

    filename = str(cbinding(name))
    nf2 = gdal.Open(filename, gdalconst.GA_ReadOnly)
    band = nf2.GetRasterBand(1)
    mapnp = band.ReadAsArray(0, 0, nf2.RasterXSize, nf2.RasterYSize).astype(np.float64)

    if compress:
        mapC = compressArray(mapnp, name=filename)
    else:
        mapC = mapnp

    if isinstance(mapC, (np.ndarray, cp.ndarray)) and mapC.dtype == np.float64:
        mapC = mapC.astype(np.float32)

    return mapC


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


def returnBool(inBinding):
    """
    Test if parameter is a boolean and return an error message if not, and the boolean if everything is ok

    :param inBinding: parameter in settings file
    :return: boolean of inBinding
    """

    b = cbinding(inBinding)
    assert isinstance(b, bool)
    return b


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
