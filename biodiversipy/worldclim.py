import os
from pickle import FALSE
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rioxarray as rxr
import janitor

# Coordinates for Germany to subset data
coords_germany = {'lon_lower': 5.7,
                  'lat_lower': 47.1,
                  'lon_upper': 15.4,
                  'lat_upper': 55.1
                 }

# Name of directory containing input files
dirname_worldclim_input = 'wc2.1_30s_bio'

# Filename of the resulting output files with bioclimatic features and these
# appended to occurences
filename_worldclim_output = 'wc2.1_30s_bio_germany.csv'
filename_occurences_input = 'occurences.csv'
filename_occurences_output = filename_occurences_input.strip('.csv') + '_worldclim.csv'

# dummy to get worldclim
GET_WORLDCLIM = False

# set absolute paths
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path_worldclim_input = os.path.join(root, 'raw_data', dirname_worldclim_input)
path_worldclim_output = os.path.join(root, 'raw_data', filename_worldclim_output)

path_occurences_input = os.path.join(root, 'biodiversipy', 'data', 'gbif', filename_occurences_input)
path_occurences_output = os.path.join(root, 'raw_data', filename_occurences_output)


def clean_worldclim_data(file_path, plot=False, coords=False, val='val'):
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
    df.columns = ['lon_lower', 'lat_upper', val]

    # Getting upper and lower bounds of boxes
    lon_df = pd.DataFrame({'lon_lower': sorted(df['lon_lower'].unique())})
    lon_df['lon_upper'] = lon_df['lon_lower'].shift(-1)

    lat_df = pd.DataFrame({'lat_upper': sorted(df['lat_upper'].unique(), reverse=True)})
    lat_df['lat_lower'] = lat_df['lat_upper'].shift(-1)
    lat_df

    df = df.merge(lon_df, how='left')
    df = df.merge(lat_df, how='left')

    df = df[['lon_lower', 'lon_upper', 'lat_lower', 'lat_upper', val]]

    df.dropna(inplace=True)

    return df


def merge_worldclim_data(dir_path='../raw_data/wc2.1_30s_bio/', coords=False):
    '''
    Wrapper for get_worldclim_data(). Given a directory, it cleans and merges
    all datasets in that directory.
    Description of each bioclimatic variable can be found here: https://worldclim.org/data/bioclim.html
    '''
    # get all files in directory
    files = os.listdir(dir_path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    data = {}

    # clean each dataset
    for file in files:
        print(file)
        file_name = os.path.join(dir_path, file)
        val = re.findall('bio_\d+', file)[0]
        df = clean_worldclim_data(file_name, plot=False, coords=coords, val=val)
        data[val] = df

    # merge datasets
    i = 0
    for key in data:
        if i == 0:
            df = data[key]
        else:
            df = df.merge(data[key], how='inner')

        i += 1

    return df

def append_worldclim_features(occurences, csv=True, bioclim_path='../raw_data/wc2.1_30s_bio_germany.csv'):
    '''
    Appends 19 bioclimatic features to a given occurences dataset. occurences can either be a path to a
    csv-file or a dictionary containing latitude and longitude. In the latter case the csv-flag must be
    set to False
    '''
    if csv:
        occurences = pd.read_csv(occurences)
    else:
        occurences = pd.DataFrame(occurences)

    bioclim_vars = pd.read_csv(bioclim_path)

    df = occurences.conditional_join(bioclim_vars,
                                 ('latitude', 'lat_lower', '>='),
                                 ('latitude', 'lat_upper', '<'),
                                 ('longitude', 'lon_lower', '>='),
                                 ('longitude', 'lon_upper', '<'),
                                 how='inner')

    df = df.drop(columns=['lon_lower', 'lon_upper', 'lat_lower', 'lat_upper'])

    return df


if __name__ == '__main__':
    # clean and save data from raw_data folder
    if GET_WORLDCLIM:
        worldclim_data = merge_worldclim_data(dir_path=path_worldclim_input, coords=coords_germany)
        worldclim_data.to_csv(path_worldclim_output, index=False)

    # append data to occurences and save the resulting output
    occurences_data = append_worldclim_features(path_occurences_input, csv=True, bioclim_path=path_worldclim_output)
    occurences_data.to_csv(path_occurences_output)
