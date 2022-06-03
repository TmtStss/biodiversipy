import pandas as pd
import numpy as np
import streamlit as st
from biodiversipy.params import coords_germany, coords_berlin
from biodiversipy.utils import get_features_for_coordinates, in_germany, in_berlin
from utils import get_coordinates

st.markdown("# Plant Species Near Me")

st.markdown("### Input Coordinates")

location = st.text_input("Enter location")

## Get coordinates for input address ##

latitude, longitude = get_coordinates(location)

#if in_germany(coords_germany, latitude, longitude):
#    st.write(f"({latitude}, {longitude}) inside of Germany")
if in_berlin(coords_berlin, latitude, longitude):
    st.write(f"({latitude}, {longitude}) inside of Berlin")
    df = get_features_for_coordinates(latitude, longitude)
    st.write(df)

else:
    #st.write(f"({latitude}, {longitude}) outside of Germany")
    st.write(f"({latitude}, {longitude}) outside of Berlin")

## Get features for corresponding coordinates


st.markdown("### Output: Most probable Plant Species to encounter")
st.markdown("-- Good luck ;)")


results = pd.read_csv("raw_data/gbif/dummy_output.csv")
st.write(results.transpose().head())

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
