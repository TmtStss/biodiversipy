import os
import re
from utils import tif_to_df
from params import coords_germany

# Name of directory containing input files
input_dir = 'wc2.1_30s_bio'

# Filename of the resulting output
file_name_out = 'wc2.1_30s_bio_germany.csv'

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
        df = tif_to_df(file_name, plot=False, coords=coords, val=val)
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

if __name__ == '__main__':
    # set absolute paths
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_data = os.path.join(root, 'raw_data', input_dir)

    # clean and save data from raw_data folder
    data = merge_worldclim_data(dir_path=path_data, coords=coords_germany)

    path_output = os.path.join(root, 'raw_data')
    data.to_csv(os.path.join(path_output, file_name_out), index=False)
