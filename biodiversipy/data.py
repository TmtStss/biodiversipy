import pandas as pd

from os import path
from sys import argv, exit

from biodiversipy.utils import merge_dfs, append_features
from biodiversipy.config import data_sources
from biodiversipy.params import coords_germany

raw_data_path = path.join(path.dirname(__file__), '..', 'raw_data')
data_path = path.join(path.dirname(__file__), 'data', 'occurences_1k_features.csv')
occurrences_file = 'occurences_1k'
occurrences_path = path.join(raw_data_path, 'gbif', occurrences_file + '.csv')

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
    print(f"Appending features to occurrences for {source['name']}...")
    feature_filename = f"{source['id']}_germany.csv"
    feature_path = path.join(raw_data_path, 'output', 'features', feature_filename)

    if not path.isfile(feature_path):
        print(f"Could not find feature file named '{feature_path}'. \nExiting")
        exit(0)

    data = append_features(occurrences_path, feature_path, from_csv)

    if (to_csv):
        occurrences_output_filename = f"{occurrences_file}_{source['name']}_germany.csv"
        occurrences_output_path = path.join(raw_data_path, 'output', 'occurrences', occurrences_output_filename)
        data.to_csv(occurrences_output_path, index=False)

    return data

def get_complete_occurrences(to_csv=True):
    '''
    Merge each of dataset specified in the config.py file containing occurrences
    and different features into one big dataset.
    '''
    for i, key in enumerate(data_sources.keys()):
        input_path = path.join(raw_data_path, 'output', 'occurrences', f"occurrences_{data_sources[key]['name']}_germany.csv")
        if i == 0:
            df = pd.read_csv(input_path)
        else:
            df_tmp = pd.read_csv(input_path)
            df = df.merge(df_tmp, how='inner')

    if to_csv:
        output_path = path.join(raw_data_path, 'output', 'occurrences', occurrences_file + '_features.csv')
        df.to_csv(output_path, index=False)

    return df

X_COLUMNS = ['a', 'list', 'of', 'column', 'names']
Y_COLUMNS = ['another', 'list', 'of', 'column', 'names']

def get_data():
    df = pd.read_csv(data_path)
    X_train = df[X_COLUMNS]
    y_train = df[Y_COLUMNS]

    return df, X_train, y_train

if __name__ == '__main__':

    if len(argv) == 1:
        for key, source in data_sources.items():
            get_tif_data(source)
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
