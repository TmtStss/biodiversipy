import joblib
from os import path

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2

LOCAL_MODEL_PATH = path.join(path.dirname(__file__), '..', 'model.joblib')

def init_model(X, y, metrics):
    normalization_layer = Normalization()
    normalization_layer.adapt(X)

    model = Sequential([
        normalization_layer,
        Dense(10,
              input_dim=X.shape[1],
              activation='relu',
              activity_regularizer=l1_l2(l1=0.005, l2=0.0005)),
        Dropout(0.4),
        Dense(y.shape[1], activation='softmax'),
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model

def load_model():
    pipeline = joblib.load(LOCAL_MODEL_PATH)
    return pipeline
