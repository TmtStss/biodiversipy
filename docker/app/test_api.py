from google.cloud import storage

import pandas as pd
from fastapi import FastAPI, Request


storage_client = storage.Client()

bucket_name = 'target_names'
data_name = '/target_names.csv'

bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(data_name)
path = 'gs://' + bucket_name + data_name

df = pd.read_csv(path)




app = FastAPI()

@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict")
def predict(request: Request):
    params = request.query_params
    return {df}
