import pandas as pd
from os import path
from sys import argv, exit
from biodiversipy.utils import merge_dfs, append_features
from biodiversipy.config import data_sources
from biodiversipy.params import coords_germany

raw_data_path = path.join(path.dirname(__file__), '..', 'raw_data')
occurrences_path = path.join(raw_data_path, 'gbif', 'occurences_1k.csv')

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

    full_data = append_features(occurrences_path, data, from_csv)

    if (to_csv):
        feature_output_filename = f"{source['id']}_germany.csv"
        feature_output_path = path.join(raw_data_path, 'output', 'features', feature_output_filename)
        data.to_csv(feature_output_path, index=False)

        occurrences_output_filename = f"occurrences_{source['name']}_germany.csv"
        occurrences_output_path = path.join(raw_data_path, 'output', 'occurrences', occurrences_output_filename)
        full_data.to_csv(occurrences_output_path, index=False)

    return full_data

def get_complete_occurrences(to_csv=True):
    '''
    Function that merges each of dataset specified in the config.py file
    containing occurrences and different features into one big dataset.
    '''
    for i, key in enumerate(data_sources.keys()):
        input_path = path.join(raw_data_path, 'output', 'occurrences', f"occurrences_{data_sources[key]['name']}_germany.csv")
        if i == 0:
            df = pd.read_csv(input_path)
        else:
            df_tmp = pd.read_csv(input_path)
            df = df.merge(df_tmp, how='inner', on=['gbifID', 'latitude', 'longitude', 'scientificName'])

    if to_csv:
        output_filename = occurrences_path.split('/')[-1].split('.csv')[0]
        output_path = path.join(raw_data_path, 'output', 'occurrences', output_filename + '_features.csv')
        df.to_csv(output_path, index=False)

    return df

if __name__ == '__main__':

    if len(argv) == 1:
        for source in data_sources.values():
            get_tif_data(source)

        exit(0)

    source_name = argv[1]

    if source_name not in data_sources.keys():
        print(f"'{source_name}' received. Expected one of {data_sources.keys()}")
        exit(0)

    get_tif_data(data_sources[source_name])
    get_complete_occurrences(to_csv=True)
