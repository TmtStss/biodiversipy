import requests

#Standard

import numpy as np
import pandas as pd

#API

from icrawler.builtin import GoogleImageCrawler
import requests


# Gets coordinates from an input location

def get_coordinates(location):
    url = "https://nominatim.openstreetmap.org/search?"
    params = {"q": location, "format": "json"}
    response = requests.get(url, params=params).json()[0]
    latitude = response["lat"]
    longitude = response["lon"]

    return float(latitude), float(longitude)
    return latitude, longitude

# 1. Get feature data from lat and long


# 2. Predict most likely taxonKeys from model




# Returns species name from a taxonKey and metadata dataframe
def species_name_from_taxonKey(taxonKey, metadata_df):

    pdseries = metadata_df.loc[metadata_df['taxonKey'] == taxonKey]['scientificName']

    return pdseries.iloc[0]

# Returns image of species from a taxonKey and metadata dataframe
def image_from_taxonKey(taxonKey, metadata_df):

    species_name = species_name_from_taxonKey(taxonKey, metadata_df)

    google_Crawler = GoogleImageCrawler(storage = {'root_dir': r'output_image'})

    google_Crawler.crawl(keyword = species_name, max_num = 1)

# Loops through taxonKeys and prints 1 pic for each
def list_of_species(list_of_taxonKeys, metadata_df):

    for taxonKey in list_of_taxonKeys:
        image_from_taxonKey(taxonKey, metadata_df)
