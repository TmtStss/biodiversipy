import joblib
import mlflow
import multiprocessing

import warnings

from google.cloud import storage
from sklearn.model_selection import train_test_split

from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from termcolor import colored
from tensorflow.keras.callbacks import EarlyStopping
from time import time

from biodiversipy.metrics import custom_metric
from biodiversipy.model import init_model
from biodiversipy.params import MLFLOW_EXPERIMENT_BASE, MLFLOW_URI
from biodiversipy.utils import simple_time_tracker, get_ranking
from biodiversipy.data import get_data, get_data_from_gcp, remove_masks

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -
BUCKET_NAME = "wagon-data-871-biodiversipy"
BUCKET_TRAIN_DATA_PATH = "xxx"

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = "biodiversipy"

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = "v1"

# Model storage location
STORAGE_LOCATION = "models/biodiversipy/model.joblib"

class Trainer(object):
    VERSION = "0.1"

    def __init__(self, X_train, y_train, **kwargs):
        """
            X: ndarray
            y: ndarray
        """
        # self.kwargs = kwargs
        # self.local = kwargs.get("local", False)  # if True training is done locally
        # self.mlflow = kwargs.get("mlflow", False)  # if True log info to mlflow
        # self.version = self.kwargs.get("version", self.VERSION)
        self.X_train, self.y_train = X_train, y_train

        # MLFlow
        # self.mf_experiment_name = f"{MLFLOW_EXPERIMENT_BASE} v{self.version}"
        # self.log_kwargs_params()
        # self.log_machine_specs()




    # @simple_time_tracker
    def train(self):
        # tic = time()
        es = EarlyStopping(patience=4)

        self.model = init_model(self.X_train, self.y_train)
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=1000,
            batch_size=16,
            callbacks=[es],
            validation_split=0.2,
            verbose=1)

        # # log to MLFlow
        # self.mlflow_log_metric("train_time", int(time() - tic))

    def evaluate(self, X_test, y_test):
        self.loss, self.eval_metric = self.model.evaluate(X_test, y_test)
        print(colored(f"loss: {self.loss} || eval metric: {self.eval_metric}", "blue"))

    def score_with_custom_metric(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        t_min, average, custom_accuracy = custom_metric(y_test, y_pred, K = 1, rate = 0.95, t = 0.01)
        print(colored(f'Model custom accuracy  = {custom_accuracy}', "blue"))
        print(colored(f'Model average ranking = {get_ranking(y_test, y_pred)}', "blue"))

    def get_baseline_model(encoded_y_df):
        '''
        Takes a one-hot-encoded y,
        predicts the probability of each species as a proportion of total species
        '''
        new_df = encoded_y_df.copy()
        for column in new_df.columns:
            new_df[column] = encoded_y_df[column].sum()/encoded_y_df.sum().sum()
        return new_df

    def baseline_score_with_custom_metric(self, y_test, y_base):
        t_min, average, custom_accuracy = custom_metric(y_test, y_base, K = 1, rate = 0.95, t = 0.01)
        print(colored(f'Baseline custom accuracy  = {custom_accuracy}', "blue"))
        print(colored(f'Baseline average ranking = {get_ranking(y_test, y_base)}', "blue"))

    def upload_model_to_gcp():
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename("model.joblib")

    def save_model(self):
        """Save model to a .joblib file"""
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

        self.upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


    # def compute_k(y_only_relevant_columns):
    #     '''
    #     only include columns with animal species, no lang/lat etc. Perhaps use:
    #     y[y.columns[2:-1]]
    #     '''
    #     return y_only_relevant_columns.sum(axis=1).mean().round().astype(int)

    # def get_K_most_probable(y_pred_proba, k):

    #     only_highest = y_pred_proba.stack().groupby(level=0).nlargest(k).unstack().reset_index(level=1, drop=True).reindex(columns=y_pred_proba.columns)
    #     fillna_highest =  only_highest.fillna(0)

    #     return fillna_highest.astype(bool).astype(int)


    ### MLFlow methods
    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.mf_experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(self.mf_experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     if self.mlflow:
    #         self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     if self.mlflow:
    #         self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    # def log_kwargs_params(self):
    #     if self.mlflow:
    #         for k, v in self.kwargs.items():
    #             self.mlflow_log_param(k, v)

    # def log_machine_specs(self):
    #     cpus = multiprocessing.cpu_count()
    #     mem = virtual_memory()
    #     ram = int(mem.total / 1000000000)
    #     self.mlflow_log_param("ram", ram)
    #     self.mlflow_log_param("cpus", cpus)

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    print(colored("## Loading data ##", "blue"))
    _, (X, y) = get_data_from_gcp(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}/features.csv",f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}/targets.csv")


    X, y = remove_masks(X, y)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    default_params = {'local': False,}

    t = Trainer(X_train, y_train, **default_params)

    print(colored("## Training model... ##", "gree"))
    t.train()

    print(colored("## Model score with custom metric... ##", "green"))
    t.score_with_custom_metric(X_test, y_test)

    print(colored("## Baseline score with custom metric... ##", "green"))
    t.score_with_custom_metric(X_test, y_test)

    print(colored("## Saving model... ##", "green"))
    t.save_model()
