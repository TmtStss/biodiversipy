import pandas as pd

from os import path, mkdir
from sys import argv, exit

from biodiversipy.utils import merge_dfs, append_features, clean_occurrences, encode_taxonKey, get_suffix
from biodiversipy.config import data_sources
from biodiversipy.params import coords_germany

# Number of samples
N = 100_000
num_species = 100

# File paths
raw_data_path = path.join(path.dirname(__file__), '..', 'raw_data')
data_path = path.join(path.dirname(__file__), 'data', 'occurrences_1k_features.csv')

occurrences_file = 'coordinates' + get_suffix(N, num_species)
occurrences_path = path.join(raw_data_path, 'gbif', \
    occurrences_file.replace('coordinates', 'occurrences'), occurrences_file + '.csv')


def get_gbif_data(csv_file='germany.csv', n=N, num_species = num_species):
    '''Clean and encode raw gbif data'''
    print(f"Cleaning occurrences data...")
    clean_occurrences(raw_data_path, csv_file, n, num_species, coords_germany)

    print(f"Encoding taxonKey...")
    merged, coordinates = encode_taxonKey(raw_data_path, n, num_species, from_csv = True, to_csv = True)

    assert len(coordinates) <= N

    return merged, coordinates

features_path = path.join(path.dirname(__file__), 'data', 'coordinates_1k_features.csv')
target_path = path.join(path.dirname(__file__), 'data', 'occurences_1k_encoded.csv')

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
            df = df.merge(df_tmp, how='inner')

    if to_csv:
        output_path = path.join(raw_data_path, 'output', 'occurrences', \
            occurrences_file, occurrences_file + '_features.csv')

        if not path.isdir(output_path):
            mkdir(output_path)

        df.to_csv(output_path, index=False)

    return df

def get_data():
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).drop(columns=['longitude', 'latitude'])

    return (X, y), (X.to_numpy(), y.to_numpy())

if __name__ == '__main__':
    # get_gbif_data(csv_file='germany.csv', n=N, num_species=num_species)

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
