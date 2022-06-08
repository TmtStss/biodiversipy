from google.cloud import storage
import joblib
from io import BytesIO
import pandas as pd
import numpy as np
from fastapi import FastAPI



# Get the Model

storage_client = storage.Client()

bucket_name = 'model-joblib'
model_name = 'test_model.joblib'

bucket = storage_client.get_bucket(bucket_name)


blob = bucket.blob(model_name)

model_file = BytesIO()

blob.download_to_file(model_file)

model=joblib.load(model_file)

# Get X_test

df = pd.read_csv('gs://wagon-data-871-biodiversipy/data/X_test_1', sep=",").drop(columns = 'Unnamed: 0')

#model.predict

output = model.predict(df)

api_output = output.tolist()

# FastAPI
app = FastAPI()

@app.get("/")
def index():
    return {"ok": api_output}
