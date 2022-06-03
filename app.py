import pandas as pd
import numpy as np
import streamlit as st
from biodiversipy.params import coords_germany

st.markdown("# Plant Species Near Me")

st.markdown("### Input Coordinates")

latitude = st.number_input(
    label="Latitude",
    step=1.0,
    format="%.6f",
    min_value=coords_germany["lat_lower"],
    max_value=coords_germany["lat_upper"],
    value=(coords_germany["lat_lower"] + coords_germany["lat_upper"]) / 2,
)

longitude = st.number_input(
    label="Longitude",
    step=1.0,
    format="%.6f",
    min_value=coords_germany["lon_lower"],
    max_value=coords_germany["lon_upper"],
    value=(coords_germany["lon_lower"] + coords_germany["lon_upper"]) / 2,
)

st.markdown("### Output: Most probable Plant Species to encounter")
st.markdown("-- Good luck ;)")


results = pd.read_csv("raw_data/gbif/dummy_output.csv")
st.write(results.transpose().head())
