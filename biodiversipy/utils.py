#Standard

from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import path
from time import time
from biodiversipy.params import coords_germany
from sklearn.feature_extraction.text import CountVectorizer

#RXR

import rioxarray as rxr
import janitor

def simple_time_tracker(method):
    '''Time tracking decorator'''
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed

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

def get_suffix(n):
    if n < 1_000:
        suffix = '_' + str(n)
    elif (n >= 1_000) and (n < 1_000_000):
        suffix = '_' + str(n // 1_000) + 'k'
    else:
        suffix = '_' + str(n // 1_000_000) + 'm'
    return suffix

def clean_occurrences(raw_data_path, csv='germany.csv', n = 0, coords=False):
    """Cleans a csv as downloaded from GBIF. Samples n rows. Outputs 2 csv files (occurrences and metadata)."""
    source_path = path.join(raw_data_path, 'gbif', csv)

    # Load data into pd.DataFrame
    data = pd.read_csv(source_path, sep = '\t', low_memory = False)

    # Keep useful columns
    selected_columns = ['gbifID', 'datasetKey', 'kingdom', 'phylum', 'class','order', 'family',
                        'genus', 'species', 'scientificName', 'decimalLatitude', 'decimalLongitude',
                        'day', 'month', 'year', 'taxonKey', 'license']

    data = data[selected_columns]

    # Drop duplicates based on lat, lon, taxonKey
    data = data.drop_duplicates(subset = ['decimalLatitude', 'decimalLongitude', 'taxonKey'], keep = 'first')

    # Rename coordinates column
    data = data.rename(columns = {'decimalLatitude': 'latitude', 'decimalLongitude': 'longitude'})

    # Drop observations outside the bounding box coordinates of Germany
    if coords:
        mask = (data['latitude'] >= coords['lat_lower']) & \
               (data['latitude'] <= coords['lat_upper']) & \
               (data['longitude'] >= coords['lon_lower']) & \
               (data['longitude'] <= coords['lon_upper'])

        data = data[mask]

    # Sample n rows
    suffix = ''
    if n:
        data = data.sample(n, random_state=1)
        suffix = get_suffix(n)

    # Splitting occurrences data and metadata
    gbifID = ['gbifID']
    taxonKey = ['taxonKey']
    coordinates = ['latitude', 'longitude']
    data_final = data[gbifID + coordinates + taxonKey]
    metadata = data.drop(columns = coordinates)

    # Create output directory
    output_path = path.join(raw_data_path,'gbif', 'occurrences' + suffix)
    if not path.isdir(output_path):
        os.mkdir(output_path)

    # Write occurrences csv
    filename = 'occurrences' + suffix + '.csv'
    destination_path = path.join(output_path, filename)
    data_final.to_csv(destination_path, index=False)

    # Write metadata csv
    filename = 'metadata' + suffix + '.csv'
    destination_path = path.join(output_path, filename)
    metadata.to_csv(destination_path, index=False)

    return data_final, metadata


def append_features(occurrences_path, features_path, from_csv=True):
    '''
    Appends features to a given occurrences dataset.
    occurrences can either be a path to a csv-file or a dictionary containing
    latitude and longitude. In the latter case the csv-flag must be set to False
    '''
    if from_csv:
        occurrences = pd.read_csv(occurrences_path)
        features = pd.read_csv(features_path)
    else:
        occurrences = pd.DataFrame(occurrences_path)
        features = pd.DataFrame(features_path)

    df = occurrences.conditional_join(features,
                                 ('latitude', 'lat_lower', '>='),
                                 ('latitude', 'lat_upper', '<'),
                                 ('longitude', 'lon_lower', '>='),
                                 ('longitude', 'lon_upper', '<'),
                                 how='inner')

    df = df.drop(columns=['lon_lower', 'lon_upper', 'lat_lower', 'lat_upper'])

    return df

def encode_taxonKey(raw_data_path, n, from_csv = True, to_csv = True):
    """
    Takes an occurence DataFrame or 'occurrences_n.csv' as input and outputs
    the species encoded and the unique location coordinates as DataFrame or
    csv ('occurrences_n_encoded.csv', 'coordinates_n.csv')
    """
    filename = 'occurrences' + get_suffix(n) + '.csv'
    source_path = path.join(raw_data_path, 'gbif', 'occurrences' + get_suffix(n), filename)

    if from_csv:
        coordinates = pd.read_csv(source_path)
    else:
        coordinates = pd.DataFrame(source_path)

    # Create a DataFrame with a coordinates column (latitude, longitude)
    coordinates['coordinates'] = coordinates[['latitude', 'longitude']].apply(tuple, axis=1)

    # Convert taxonKey to string for later vectorizing
    coordinates['taxonKey'] = coordinates['taxonKey'].astype('string')

    # Group by coordinates and list the taxonKey's
    encoded_targets = coordinates.groupby(['coordinates'])['taxonKey'].apply(list)
    encoded_targets = pd.DataFrame(encoded_targets)
    idx = encoded_targets.index

    # Format taxonKey Pandas Series for vectorizing
    encoded_targets['taxonKey'] = encoded_targets['taxonKey'].map(lambda x: ' '.join(x))

    # Initialize CountVectorizer and apply it to the taxonKey's
    vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split())
    encoded_targets = vectorizer.fit_transform(encoded_targets['taxonKey']).toarray()

    # Get feature names out
    encoded_targets = pd.DataFrame(encoded_targets, index=idx, columns = vectorizer.get_feature_names_out())
    encoded_targets.reset_index(inplace=True)

    # Merging output of CountVectorizer with latitude and longitude data
    coordinates = coordinates.drop(columns=['gbifID', 'taxonKey']).drop_duplicates()
    merged = coordinates.merge(encoded_targets).drop(columns='coordinates')
    coordinates = coordinates.drop(columns='coordinates')

    if to_csv:
        encoded_path = source_path.replace('.csv', '_encoded.csv')
        merged.to_csv(encoded_path, index = False)

        coordinates_filename = filename.replace('occurrences', 'coordinates')
        coordinates_path = path.join(raw_data_path, 'gbif', 'occurrences' + get_suffix(n), coordinates_filename)
        coordinates.to_csv(coordinates_path, index = False)

    return merged, coordinates

import sys

if __name__ == "__main__":
    _, csv, n = sys.argv
    clean_occurrences('biodiversipy/../raw_data/', csv, int(n))
    print('step 1 done')
    encode_taxonKey('biodiversipy/../raw_data', int(n))
    print('step 2 done')
