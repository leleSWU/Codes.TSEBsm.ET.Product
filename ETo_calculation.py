from osgeo import gdal, osr
import netCDF4
import math
import os
import numpy as np
import glob
import datetime
import matplotlib.pyplot as plt
from time import *


def main():
    year = 2016
    daily_file_d = r"Q:\G\Data_Input_instantaneous_daily_1°"
    t2m_max_min_d = r"H:\G\Data_EToF_Daily"
    DNN_daily_ET = r"P:\G\Output\2016_new\DNN_ERA5_all_1000\LE"
    out_d = r'P:\G\Output\2016_new\EToF_ERA5'
    ssrd_list = np.array(glob.glob(r"Q:\G\Data_Input_instantaneous_daily_1°\ssrd\2010" + "\*.tiff"))

    t2m_list = np.array(glob.glob(daily_file_d + "\\t2m\\" + str(year) + "\*.tiff"))
    albedo_list = np.array(glob.glob(daily_file_d + "\\albedo\\" + str(year) + "\*.tiff"))
    SSR_list = np.array(glob.glob(daily_file_d + "\\ssrd\\" + str(year) + "\*.tiff"))
    t2m_max_list = np.array(glob.glob(t2m_max_min_d + "\\t2m_max_min\\" + str(year) + "\*Max.tiff"))
    t2m_min_list = np.array(glob.glob(t2m_max_min_d + "\\t2m_max_min\\" + str(year) + "\*Min.tiff"))
    r_list = np.array(glob.glob(daily_file_d + "\\r\\" + str(year) + "\*.tiff"))
    sp_list = np.array(glob.glob(daily_file_d + "\\sp\\" + str(year) + "\*.tiff"))
    ws_list = np.array(glob.glob(daily_file_d + "\\ws\\" + str(year) + "\*.tif"))

    DNN_daily_ET_list = np.array(glob.glob(DNN_daily_ET + "\*.tif"))
    DEN_file = r"Q:\G\Data_Input_instantaneous_daily_1°\DEM\DEM_region.tif"
    lat_file = r"Q:\G\Data_Input_instantaneous_daily_1°\lat\lat_pre_73.0_55.0_135.0_16.0.tif"
    DEM = gdal.Open(DEN_file).ReadAsArray()
    lat = gdal.Open(lat_file).ReadAsArray()

    for i in range(0, len(t2m_list)):
        doy = i + 1
        print(year, doy)
        geoTran = gdal.Open(t2m_list[i]).GetGeoTransform()
        Ta = gdal.Open(t2m_list[i]).ReadAsArray() - 273.15
        Albedo = gdal.Open(albedo_list[i]).ReadAsArray() / 100
        Rs = gdal.Open(SSR_list[i]).ReadAsArray()
        Ta_max = gdal.Open(t2m_max_list[i]).ReadAsArray() - 273.15
        Ta_min = gdal.Open(t2m_min_list[i]).ReadAsArray() - 273.15
        rh = gdal.Open(r_list[i]).ReadAsArray() / 100
        pres = gdal.Open(sp_list[i]).ReadAsArray() * 100
        u2 = gdal.Open(ws_list[i]).ReadAsArray()

        deta = 4098 * (0.6108 * np.exp(17.27 * Ta / (Ta + 237.3))) / (Ta + 237.3) ** 2
        Rns = (1 - Albedo) * Rs
        TamaxK = Ta_max + 273.15
        TaminK = Ta_min + 273.15

        es_max = 0.61808 * np.exp(17.27 * Ta_max / (Ta_max + 237.3))
        es_min = 0.61808 * np.exp(17.27 * Ta_max / (Ta_min + 237.3))
        es = (es_max + es_min) / 2
        ea = rh * es

        J = doy
        dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)
        Gsc = 0.0820
        Afi = lat * np.pi / 180
        Athi = 0.408 * np.sin(2 * np.pi / 365 * J - 1.39)
        ws = np.arccos(-np.tan(Afi) * np.tan(Athi))
        Ra = 24 * 60 * Gsc * dr * (ws * np.sin(Afi) * np.sin(Athi) + np.cos(Afi) * np.cos(Athi) * np.sin(ws)) / np.pi
        sigma = 4.903 * (10E-9)
        Rso = (0.75 + 2 * 10 ** 5 * DEM) * Ra
        Rnl = sigma * ((TamaxK ** 4 + TaminK ** 4) / 2) * (0.34 - 0.14 * ea ** 0.5) * (1.35 * Rs / Rso - 0.35)
        Rn = Rns - Rnl
        FIFI = 0.665 * (10E-3) * pres / 100
        Tmean = (Ta_max + Ta_min) / 2
        G0 = 0.0
        PE = (0.408 * deta * (Rn - G0) + FIFI * 900 * u2 * (es - ea) / (Tmean + 273.15)) / (
                deta + FIFI * (1 + 0.34 * u2))
        PE_OK = PE
        PE_OK[PE_OK > 2000] = 2000
        PE_OK[PE_OK < 0] = 0
        PE_OK[np.isnan(PE_OK)] = 0

        ssrd_data = gdal.Open(ssrd_list[i]).ReadAsArray()
        Nodata_value = 0

        out_name = out_d + "\\ET0\\ET0" + str(year) + str(doy).zfill(3) + "_ERA5.tif"
        Write_Tiff_optimize(out_name, PE_OK, Ta.shape[1], Ta.shape[0], geoTran[0], geoTran[3], geoTran[1], Nodata_value)

        print(1)


def Write_Tiff_optimize(outname, data, X_size, Y_size, lon_min, lat_max, resolution, NoData_value=None):
    import os
    (filepath, tempfilename) = os.path.split(outname)
    if os.path.exists(filepath) == False:
        os.makedirs(filepath)
    ds = gdal.GetDriverByName('Gtiff').Create(outname, int(X_size), int(Y_size), 1, gdal.GDT_Float32)
    geotransfrom = (lon_min, resolution, 0, lat_max, 0, -resolution)
    ds.SetGeoTransform(geotransfrom)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(data)
    if NoData_value != None:
        ds.GetRasterBand(1).SetNoDataValue(NoData_value)
    ds.FlushCache()
    ds = None


if __name__ == '__main__':
    main()
