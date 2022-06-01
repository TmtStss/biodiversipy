from biodiversipy.params import coords_germany

#Standard

from urllib.error import HTTPError
import numpy as np

# SoilGrid

from owslib.wcs import WebCoverageService

# OS

from os import path

#data path

raw_data_path = path.join(path.dirname(__file__), '..', 'raw_data','soilgrid_tiffs')

# Function to download data in seperate tiff files:

def data_collector(properties_list):

    '''
    Function for downloading tiff files for every layer of every soilgrid feature.
    Returns nothing.
    Takes 20 minutes to run on my laptop.
    '''

    # Define resX and resY

    phi = np.cos(50/180)

    a = 6378137
    b = 6356752

    resX = 250 / (2*np.pi*a/360) / np.cos(phi)
    resY = 250 / (2*np.pi*(a + b) / 2 / 360)

    # Germany bbox

    lon_min = coords_germany['lon_lower']
    lat_min = coords_germany['lat_lower']
    lon_max = coords_germany['lon_upper']
    lat_max = coords_germany['lat_upper']

    bbox = (lon_min, lat_min, lon_max, lat_max)

    # Loop over every property that we want

    for prop in properties_list:

        # Connect to the server on that property

        wcs = WebCoverageService(f"http://maps.isric.org/mapserv?map=/map/{prop}.map", version='1.0.0')

        # Get each layer for that property

        list_of_keys = list(wcs.contents.keys())

        # Get only the mean values for each layer (and not the quartiles, median, etc.)

        mean_list = [key for key in list_of_keys if key.endswith('mean')]

        # downloads data

        for layer in mean_list:

            print (layer + ' attempting download')

            # See if its already downloaded before an error occured:
            layer_path = path.join(raw_data_path, layer + '.tif')

            if not path.exists(layer_path):

                # Calls the server
                try:
                    response = wcs.getCoverage(
                        identifier = layer,
                        crs='urn:ogc:def:crs:EPSG::4326',
                        bbox= bbox,
                        resx=resX, resy=resY,
                        format='GEOTIFF_INT16')

                except ConnectionError or HTTPError:
                    data_collector(properties_list)

                # Downloads the data

                with open(layer_path, 'wb') as file:
                    file.write(response.read())

            print (layer + ' downloaded successfully')


properties_list = ['bdod', 'cec', 'cfvo', 'clay', 'nitrogen', 'phh2o', 'sand', 'silt', 'soc', 'ocd', 'ocs']

data_collector(properties_list)
