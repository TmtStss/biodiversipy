import pandas as pd
import numpy as np

from os import path, mkdir, listdir
from sys import argv, exit
import re

from biodiversipy.utils import merge_dfs, append_features, append_features_split, clean_occurrences, encode_taxonKey, get_suffix
from biodiversipy.config import data_sources
from biodiversipy.params import coords_germany

from scipy.sparse import load_npz

# Number of samples
N = False
num_species = False

# File paths
raw_data_path = path.join(path.dirname(__file__), '..', 'raw_data')
data_path = path.join(path.dirname(__file__), 'data', 'occurrences_1k_features.csv')
features_path = path.join(
    raw_data_path,
    'output/occurrences/coordinates_100k/coordinates_100k_features.csv')
target_path = path.join(
    raw_data_path,
    'gbif/occurrences_100k/occurrences_100k_encoded.csv')

occurrences_file = 'coordinates' + get_suffix(N, num_species)
occurrences_path = path.join(raw_data_path, 'gbif', \
    occurrences_file.replace('coordinates', 'occurrences'), occurrences_file + '.csv')


def get_gbif_data(csv_file='germany.csv', n=N, num_species = num_species, sparse=False):
    '''Clean and encode raw gbif data'''
    print(f"Cleaning occurrences data...")
    clean_occurrences(raw_data_path, csv_file, n, num_species, coords_germany)

    print(f"Encoding taxonKey...")
    merged, coordinates = encode_taxonKey(raw_data_path, n, num_species, from_csv = True, to_csv = True, sparse=sparse)

    return merged, coordinates

def get_tif_data(source, to_csv=True, from_csv=True):
    '''Extract data from tif files'''
    print(f"Extracting data for {source['name']}...")
    source_path = path.join(raw_data_path, source['id'])

    if not path.isdir(source_path):
        print(f"Could not find directory named '{source['id']}'. \nExiting")
        exit(0)

    data = merge_dfs(
        source_path=source_path,
        coords=coords_germany,
        file_sort_fn=source['file_sort_fn'],
        column_name_extractor=source['column_name_extractor'])

    if (to_csv):
        feature_output_filename = f"{source['id']}_germany.csv"
        feature_output_path = path.join(raw_data_path, 'output', 'features', feature_output_filename)
        data.to_csv(feature_output_path, index=False)

    return data

def get_features(source, to_csv=True, from_csv=True):
    '''Append features to occurrences'''
    print(f"Appending {source['name']} features to occurrences...")
    feature_filename = f"{source['id']}_germany.csv"
    feature_path = path.join(raw_data_path, 'output', 'features', feature_filename)

    if not path.isfile(feature_path):
        print(f"Could not find feature file named '{feature_path}'. \nExiting")
        exit(0)

    data = append_features(occurrences_path, feature_path, from_csv)

    if (to_csv):
        occurrences_output_filename = f"{occurrences_file}_{source['name']}_germany.csv"
        occurrences_output_path = path.join(raw_data_path, 'output', 'occurrences', \
            occurrences_file)

        if not path.isdir(occurrences_output_path):
            mkdir(occurrences_output_path)

        data.to_csv(path.join(occurrences_output_path, occurrences_output_filename), index=False)

    return data

def split_occurrences(n_splits=5):
    assert not N
    print(f"Splitting data into {n_splits}")

    folder = path.join(raw_data_path, 'gbif', 'occurrences')

    if not path.isdir(path.join(folder, 'splits')):
            mkdir(path.join(folder, 'splits'))

    data = pd.read_csv(path.join(folder, 'coordinates.csv'))

    length_subsample = int(len(data) / n_splits + 1)

    for i in range(n_splits):
        start = length_subsample * i
        end = length_subsample * (i + 1)
        subsample = data[start:end]
        subsample.to_csv(path.join(folder, 'splits', f'coordinates_{i + 1}.csv'), index=False)

def get_features_split(source, begin=1):
    '''Append features to occurrences'''
    assert not N
    print(f"Appending {source['name']} features to occurrences...")
    feature_filename = f"{source['id']}_germany.csv"
    feature_path = path.join(raw_data_path, 'output', 'features', feature_filename)

    if not path.isfile(feature_path):
        print(f"Could not find feature file named '{feature_path}'. \nExiting")
        exit(0)

    features = pd.read_csv(feature_path)
    splits_path = path.join(raw_data_path, 'gbif', 'occurrences', 'splits')
    files = listdir(splits_path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    out_path = path.join(raw_data_path, 'output', 'occurrences', 'coordinates', 'splits')

    if not path.isdir(out_path):
        mkdir(out_path)

    for split in range(begin, len(files) + 1):
        print(f"Running split {split}")
        file = files[split - 1]
        data = append_features_split(path.join(splits_path, file), features)

        filename = f"{occurrences_file}_{split}_{source['name']}_germany.csv"

        data.to_csv(path.join(out_path, filename), index=False)
        del data
        print(f"Finished running split {split} for {source['name']}")

    print(f"Finished all splits for {source['name']}. Merging")
    files = [file for file in listdir(out_path) if f"_{source['name']}_" in file]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for i, file in enumerate(files):
        if i == 0:
            data = pd.read_csv(path.join(out_path, file))
        else:
            tmp = pd.read_csv(path.join(out_path, file))
            data = pd.concat((data,tmp), ignore_index=True)
            del tmp

    filename = f"{occurrences_file}_{source['name']}_germany.csv"
    data.to_csv(path.join(raw_data_path, 'output', 'occurrences', 'coordinates', filename), index=False)

def get_complete_occurrences(to_csv=True):
    '''
    Merge each of dataset specified in the config.py file containing occurrences
    and different features into one big dataset.
    '''
    print("Merging features")
    for i, key in enumerate(data_sources.keys()):
        input_path = path.join(raw_data_path, 'output', 'occurrences', occurrences_file, \
            f"{occurrences_file}_{data_sources[key]['name']}_germany.csv")
        if i == 0:
            df = pd.read_csv(input_path)
        else:
            df_tmp = pd.read_csv(input_path)
            df = df.merge(df_tmp, how='right')

    if to_csv:
        output_file = occurrences_file + '_features.csv'
        output_path = path.join(raw_data_path, 'output', 'occurrences', \
            occurrences_file)

        if not path.isdir(output_path):
            mkdir(output_path)
            
        df.fillna(-3.400000e+38, inplace=True)

        df.to_csv(path.join(output_path, occurrences_file + '_features.csv'), index=False)

    return df

def get_data(from_csv=True):
    X = pd.read_csv(features_path)
    if from_csv:
        y = pd.read_csv(target_path).drop(columns=['longitude', 'latitude'])
        return (X, y), (X.to_numpy(), y.to_numpy())
    else:
        y = load_npz(target_path).toarray()
    return (X, y), (X.to_numpy(), y)

if __name__ == '__main__':

    get_gbif_data(csv_file='germany.csv', n=N, num_species=num_species, sparse=True)


    if len(argv) == 1:
        for key, source in data_sources.items():
            #get_tif_data(source)
            get_features(source)

        get_complete_occurrences(to_csv=True)

        exit(0)

    source_name = argv[1]

    if source_name not in data_sources.keys():
        print(f"'{source_name}' received. Expected one of {data_sources.keys()}")
        exit(0)

    get_tif_data(data_sources[source_name])
    get_features(data_sources[source_name])

    for source in data_sources:
        feature_filename = f"{occurrences_file}_{data_sources[source]['name']}_germany.csv"
        feature_path = path.join(raw_data_path, 'output', 'occurrences', feature_filename)
        if not path.isfile(feature_path):
            print(f"Could not find feature file named '{feature_path}'. \nExiting")
            exit(0)
        get_complete_occurrences(to_csv=True)

    # running splits
    # split_occurrences(n_splits=25)
    # get_features_split(data_sources[source_name], begin=1)
    # get_complete_occurrences()
