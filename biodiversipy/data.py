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

    if (to_csv):
        output_filename = f"{source['id']}_germany.csv"
        output_path = path.join(raw_data_path, output_filename)
        data.to_csv(output_path, index=False)

    full_data = append_features(occurrences_path, data, from_csv)

    if (to_csv):
        output_filename = f"occurences_{source['name']}_germany.csv"
        output_path = path.join(raw_data_path, output_filename)
        full_data.to_csv(output_path, index=False)

    return full_data

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
