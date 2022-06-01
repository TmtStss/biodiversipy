import joblib
import mlflow
import multiprocessing
import pandas as pd

from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from termcolor import colored
from time import time

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from biodiversipy.encoders import SomeTransformer
from biodiversipy.params import MLFLOW_EXPERIMENT_BASE, MLFLOW_URI, ESTIMATORS
from biodiversipy.utils import get_error, simple_time_tracker

class Trainer(object):
    ESTIMATOR = "SVC" # TODO set correct default
    VERSION = "0.1"

    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.kwargs = kwargs
        self.local = kwargs.get("local", False)  # if True training is done locally
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to mlflow
        self.estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        self.Estimator = ESTIMATORS[self.estimator]
        self.version = self.kwargs.get("version", self.VERSION)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.3)

        # MLFlow
        self.mf_experiment_name = f"{MLFLOW_EXPERIMENT_BASE} {self.estimator} v{self.version}"
        self.log_estimator_params()
        self.log_kwargs_params()
        self.log_machine_specs()

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        some_pipe = Pipeline([
            ('some_transformer', SomeTransformer()),
            ('stdscaler', StandardScaler())
        ])

        some_other_pipe = Pipeline([
            ('some_other_transformer', SomeTransformer()),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
            ('some_pipe', some_pipe, ['some_column']),
            ('some_other_pipe', some_other_pipe, ['some_other_column'])
        ], remainder="drop")

        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('some_model', self.Estimator())
        ])

    @simple_time_tracker
    def train(self):
        tic = time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

        # log to MLFlow
        self.mlflow_log_metric("train_time", int(time() - tic))

    def evaluate(self):
        train_error = self.compute_error(self.X_train, self.y_train)
        self.mlflow_log_metric("train_error", train_error)

        val_error = self.compute_error(self.X_val, self.y_val, show=True)
        self.mlflow_log_metric("val_error", val_error)

        print(colored(f"train error: {train_error} || val error: {val_error}", "blue"))

    def compute_error(self, X_test, y_test, show=False):
        # TODO define custom metric

        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")

        y_pred = self.pipeline.predict(X_test)

        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))

        err = get_error(y_pred, y_test)
        return round(err, 3)

    def save_model(self):
        """Save model to a .joblib file"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

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

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

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