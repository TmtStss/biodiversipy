import joblib
import mlflow
import multiprocessing

from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from termcolor import colored
from tensorflow.keras.callbacks import EarlyStopping
from time import time

from biodiversipy.metrics import custom_metric
from biodiversipy.model import init_model
from biodiversipy.params import MLFLOW_EXPERIMENT_BASE, MLFLOW_URI
from biodiversipy.utils import simple_time_tracker

class Trainer(object):
    VERSION = "0.1"

    def __init__(self, X_train, y_train, **kwargs):
        """
            X: ndarray
            y: ndarray
        """
        self.kwargs = kwargs
        self.local = kwargs.get("local", False)  # if True training is done locally
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to mlflow
        self.version = self.kwargs.get("version", self.VERSION)
        self.X_train, self.y_train = X_train, y_train

        # MLFlow
        self.mf_experiment_name = f"{MLFLOW_EXPERIMENT_BASE} v{self.version}"
        self.log_kwargs_params()
        self.log_machine_specs()

    @simple_time_tracker
    def train(self):
        tic = time()
        es = EarlyStopping(patience=4)

        self.model = init_model(self.X_train, self.y_train, metrics=[custom_metric])
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=1000,
            batch_size=16,
            callbacks=[es],
            validation_split=0.3,
            verbose=1)

        # log to MLFlow
        self.mlflow_log_metric("train_time", int(time() - tic))

    def evaluate(self, X_test, y_test):
        self.loss, self.eval_metric = self.model.evaluate(X_test, y_test)
        print(colored(f"loss: {self.loss} || eval metric: {self.eval_metric}", "blue"))

    def save_model(self):
        """Save model to a .joblib file"""
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    def get_baseline_model(encoded_y_df):
        '''
        Takes a one-hot-encoded y,
        predicts the probability of each species as a proportion of total species
        '''
        new_df = encoded_y_df.copy()
        for column in new_df.columns:
            new_df[column] = encoded_y_df[column].sum()/encoded_y_df.sum().sum()
        return new_df

    def compute_k(y_only_relevant_columns):
        '''
        only include columns with animal species, no lang/lat etc. Perhaps use:
        y[y.columns[2:-1]]
        '''
        return y_only_relevant_columns.sum(axis=1).mean().round().astype(int)

    def get_K_most_probable(y_pred_proba, k):

        only_highest = y_pred_proba.stack().groupby(level=0).nlargest(k).unstack().reset_index(level=1, drop=True).reindex(columns=y_pred_proba.columns)
        fillna_highest =  only_highest.fillna(0)

        return fillna_highest.astype(bool).astype(int)


    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.mf_experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.mf_experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)
