from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def init_model(X, y, metrics):
    normalization_layer = Normalization()
    normalization_layer.adapt(X)

    model = Sequential([
        normalization_layer,
        Dense(10, input_dim=X.shape[1], activation='relu'),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(y.shape[1], activation='softmax'),
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[metrics])

    return model