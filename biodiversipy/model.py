import joblib
from os import path

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam

LOCAL_MODEL_PATH = path.join(path.dirname(__file__), '..', 'model.joblib')

def init_model(X, y):
    normalization_layer = Normalization()
    normalization_layer.adapt(X)
    # regularizer = l1_l2

    model = Sequential([
        normalization_layer,
        Dense(100,
              activation='relu'),
        Dropout(0.2),
        Dense(50,
              activation='relu'),
        Dropout(0.2),
        Dense(20,
              activation='tanh'),
        Dense(3615, activation='softmax'),
    ])

    lr_schedule = ExponentialDecay(0.001, decay_steps = 5000, decay_rate = 0.95)
    opt = Adam(learning_rate = lr_schedule)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    return model

def load_model():
    pipeline = joblib.load(LOCAL_MODEL_PATH)
    return pipeline
