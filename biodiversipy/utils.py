#Standard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import path
from biodiversipy.params import coords_germany

#RXR

import rioxarray as rxr
import janitor

def in_germany(coords_germany, lat, lon):
    """Returns True if the (lat,lon) are within the bounding box coordinates of Germany"""

    if lat > coords_germany['lat_upper']:
        return False
    elif lat < coords_germany['lat_lower']:
        return False
    elif lon > coords_germany['lon_upper']:
        return False
    elif lon < coords_germany['lon_lower']:
        return False
    return True

def tif_to_df(file_path, plot=False, coords=False, column_name='val'):
    '''
    Function for cleaning one of the 19 worldclim datasets.
    '''
    dataarray = rxr.open_rasterio(file_path)
    df = dataarray[0].to_pandas()

    if coords:
        # subsetting based on coords dictionary
        subset_lon = np.logical_and(df.columns >= coords['lon_lower'],
                                    df.columns <= coords['lon_upper'])

        subset_lat = np.logical_and(df.index   >= coords['lat_lower'],
                                    df.index   <= coords['lat_upper'])

        df = df.iloc[subset_lat, subset_lon]


    if plot:
        fig, ax = plt.subplots(figsize=(16, 5))

        # masking values
        df_masked = np.ma.masked_where((-273 > df), df)

        # set axis
        x_min = round((df.columns).min())
        x_max = round((df.columns).max())
        y_min = round((df.index).min())
        y_max = round((df.index).max())

        # use imshow so that we have something to map the colorbar to
        image = ax.imshow(df_masked,
                          extent=[x_min, x_max, y_min, y_max])

        # add colorbar using the now hidden image
        fig.colorbar(image, ax=ax)
        plt.show()

    df = df.unstack(level=-1)
    df = df.reset_index()
    df.columns = ['lon_lower', 'lat_upper', column_name]

    # Getting upper and lower bounds of boxes
    lon_df = pd.DataFrame({'lon_lower': sorted(df['lon_lower'].unique())})
    lon_df['lon_upper'] = lon_df['lon_lower'].shift(-1)

    lat_df = pd.DataFrame({'lat_upper': sorted(df['lat_upper'].unique(), reverse=True)})
    lat_df['lat_lower'] = lat_df['lat_upper'].shift(-1)
    lat_df

    df = df.merge(lon_df, how='left')
    df = df.merge(lat_df, how='left')

    df = df[['lon_lower', 'lon_upper', 'lat_lower', 'lat_upper', column_name]]

    df.dropna(inplace=True)

    return df

def merge_dfs(source_path, coords=False, file_sort_fn=None, column_name_extractor=lambda file: file):
    '''
    Wrapper for get_worldclim_data(). Given a directory, it cleans and merges
    all datasets in that directory.
    Description of each bioclimatic variable can be found here: https://worldclim.org/data/bioclim.html
    '''
    # get all files in directory
    files = os.listdir(source_path)
    files.sort(key=file_sort_fn)

    data = {}

    # clean each dataset
    for file in files:
        print(file)
        file_name = os.path.join(source_path, file)
        column_name = column_name_extractor(file)
        df = tif_to_df(file_name, plot=False, coords=coords, column_name=column_name)
        data[column_name] = df

    # merge datasets
    i = 0
    for key in data:
        if i == 0:
            df = data[key]
        else:
            df = df.merge(data[key], how='inner')

        i += 1

    return df

def clean_occurences(csv, n = 0):
    """Cleans a csv as downloaded from GBIF. Samples n rows. Outputs 2 csv files (occurences and metadata)."""

    # Define source path
    raw_data_path = path.join(path.dirname(__file__), '..', 'raw_data')
    source_path = path.join(raw_data_path, 'gbif', csv)

    # Load data into pd.DataFrame
    data = pd.read_csv(source_path, sep = '\t', low_memory = False)

    # Keep useful columns
    selected_columns = ['gbifID', 'datasetKey', 'kingdom', 'phylum', 'class',
       'order', 'family', 'genus', 'species', 'scientificName', 'decimalLatitude', 'decimalLongitude', 'day',
       'month', 'year', 'taxonKey', 'license']
    data_cleaned = data[selected_columns]

    # Drop duplicates based on lat, lon, scientificName
    data_cleaned = data_cleaned.drop_duplicates(subset = ['decimalLatitude', 'decimalLongitude', 'scientificName'], keep = 'first')

    # Rename coordinates column
    data_cleaned = data_cleaned.rename(columns = {'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})

    # Drop observations outside the bounding box coordinates of Germany
    mask_germany = data_cleaned.apply(lambda row: in_germany(coords_germany, row.latitude, row.longitude), axis = 1)
    data_cleaned = data_cleaned[mask_germany]

    # Sample n rows
    suffix = ''
    if n:
        data_cleaned = data_cleaned.sample(n)
        suffix = '_' + str(n)

    # Splitting occurences data and metadata
    gbifID = ['gbifID']
    col = ['latitude', 'longitude', 'scientificName']
    data_final = data_cleaned[gbifID + col]
    metadata = data_cleaned.drop(columns = col)

    # Write occurences csv
    filename = 'occurences' + suffix + '.csv'
    destination_path = path.join(raw_data_path,'gbif', filename)
    data_final.to_csv(destination_path, index=False)

    # Write metadata csv
    filename = 'metadata' + suffix + '.csv'
    destination_path = path.join(raw_data_path,'gbif', filename)
    metadata.to_csv(destination_path, index=False)

def append_features(occurences_path, features, from_csv=True):
    '''
    Appends features to a given occurences dataset.
    Occurences can either be a path to a csv-file or a dictionary containing
    latitude and longitude. In the latter case the csv-flag must be set to False
    '''
    if from_csv:
        occurences = pd.read_csv(occurences_path)
    else:
        occurences = pd.DataFrame(occurences_path)

    features = pd.DataFrame(features)

    df = occurences.conditional_join(features,
                                 ('latitude', 'lat_lower', '>='),
                                 ('latitude', 'lat_upper', '<'),
                                 ('longitude', 'lon_lower', '>='),
                                 ('longitude', 'lon_upper', '<'),
                                 how='inner')

    df = df.drop(columns=['lon_lower', 'lon_upper', 'lat_lower', 'lat_upper'])
