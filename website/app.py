'''Imports'''

from re import L
import pandas as pd
import numpy as np
import streamlit as st
from biodiversipy.params import coords_germany, coords_berlin
from biodiversipy.utils import get_features_for_coordinates, in_germany, in_berlin
from utils import get_coordinates, get_features_for_coordinates, species_name_from_taxonKey, image_from_taxonKey
from os import path
import os
from PIL import Image

st.markdown("# Plant Species Near Me")

st.markdown("### Input Coordinates")

location = st.text_input("Enter location")

## Get coordinates for input address ##

latitude, longitude = get_coordinates(location)

## Then checks if in zone and creates df of features for coordinates ##


#if in_germany(coords_germany, latitude, longitude):
#    st.write(f"({latitude}, {longitude}) inside of Germany")
if in_berlin(coords_berlin, latitude, longitude):
    st.write(f"({latitude}, {longitude}) inside of Berlin")
    df = get_features_for_coordinates(latitude, longitude)
    st.write(df)

else:
    #st.write(f"({latitude}, {longitude}) outside of Germany")
    st.write(f"({latitude}, {longitude}) outside of Berlin")



## Then run the model pipeline to output taxonKeys

# A potential taxonKey would be:
#dummy_taxonKey = int(metadata.sample(1)['taxonKey'])


## Then get species name from taxonKeys

raw_data_path = path.join(path.dirname(__file__), '..', 'raw_data')
source_path = path.join(raw_data_path, 'gbif', 'metadata_1000.csv')

metadata_df = pd.read_csv(source_path)

dummy_taxonKey = int(metadata_df.sample(1)['taxonKey'])

species_name = species_name_from_taxonKey(dummy_taxonKey, metadata_df)






st.markdown(f"### Output: Most probable Plant Species to encounter is {species_name}")

##

image_from_taxonKey(dummy_taxonKey, metadata_df)

image = Image.open('raw_data/output/images/000001.jpg')

st.image(image)

st.markdown("-- Good luck ;)")


# results = pd.read_csv("raw_data/gbif/dummy_output.csv")
# st.write(results.transpose().head())

# latitude = st.number_input(
#     label="Latitude",
#     step=1.0,
#     format="%.6f",
#     min_value=coords_germany["lat_lower"],
#     max_value=coords_germany["lat_upper"],
#     value=(coords_germany["lat_lower"] + coords_germany["lat_upper"]) / 2,
# )

# longitude = st.number_input(
#     label="Longitude",
#     step=1.0,
#     format="%.6f",
#     min_value=coords_germany["lon_lower"],
#     max_value=coords_germany["lon_upper"],
#     value=(coords_germany["lon_lower"] + coords_germany["lon_upper"]) / 2,
# )
