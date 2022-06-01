import re

data_sources = {
    'worldclim': {
        'id': 'wc2.1_30s_bio',
        'name': 'worldclim',
        'file_sort_fn': lambda f: int(''.join(filter(str.isdigit, f))),
        'column_name_extractor': lambda file: re.findall('bio_\d+', file)[0],
    },
    'gee': {
        'id': 'USGS_SRTMGL1_003',
        'name': 'gee',
        'file_sort_fn': None,
        'column_name_extractor': lambda file: re.findall('germany_(\w+).tif', file)[0],
    },
    'soilgrids': {
        'id': 'soilgrid_tiffs',
        'name': 'soilgrids',
        'file_sort_fn': None,
        'column_name_extractor': lambda file: re.findall('(^.*)(?=_mean\.tif)', file)[0]
    }
}
