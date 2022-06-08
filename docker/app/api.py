from google.cloud import storage
import joblib
from io import BytesIO
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request



# Get the Model

storage_client = storage.Client()

bucket_name = 'model-joblib'
model_name = 'test_model.joblib'

bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(model_name)
model_file = BytesIO()
blob.download_to_file(model_file)

model=joblib.load(model_file)

# Get target_names
storage_client = storage.Client()

bucket_name = 'target_names'
data_name = '/target_names.csv'

bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(data_name)
path = 'gs://' + bucket_name + data_name

target_names = pd.read_csv(path)

# FastAPI
app = FastAPI()

@app.get("/")
def index():
    return {"ok": 'positively'}

@app.get("/predict")
def predict(request: Request):

    params = request.query_params
    #params is a dictionary with all features
    df_X_test = pd.DataFrame(dict(params), index = [0]).apply(pd.to_numeric)
    #run into model
    output = model.predict(df_X_test)[0]

    target_names['probability'] = output
    out_df = target_names.sort_values('probability', ascending= False).head(5)

    output_dict = {'species': out_df.to_dict('records')}

    return output_dict
