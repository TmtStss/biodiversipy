from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

coords_germany = {
    'lon_lower': 5.7,
    'lat_lower': 47.1,
    'lon_upper': 15.4,
    'lat_upper': 55.1}

MLFLOW_EXPERIMENT_BASE = "[germany] [berlin] [biodiversipy-team] "
MLFLOW_URI = "https://mlflow.lewagon.ai/"

# just examples
ESTIMATORS = {
    'svc': SVC,
    'sgdc': SGDClassifier,
}
