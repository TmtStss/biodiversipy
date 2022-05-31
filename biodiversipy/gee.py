import os
from biodiversipy.utils import merge_dfs
from biodiversipy.params import coords_germany

# Name of directory containing input files
input_dir = 'USGS_SRTMGL1_003'

# Filename of the resulting output
file_name_out = 'USGS_SRTMGL1_003_germany.csv'

if __name__ == '__main__':
    # set absolute paths
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_data = os.path.join(root, 'raw_data', input_dir)

    # clean and save data from raw_data folder
    data = merge_dfs(
        dir_path=path_data,
        coords=coords_germany,
        column_name_extractor=lambda file: file)

    path_output = os.path.join(root, 'raw_data')
    data.to_csv(os.path.join(path_output, file_name_out), index=False)
