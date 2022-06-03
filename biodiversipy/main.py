import warnings

from termcolor import colored

from sklearn.model_selection import train_test_split

from biodiversipy.data import get_data
from biodiversipy.trainer import Trainer

default_params = {
    'local': True,
}

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    print(colored("## Loading data ##", "blue"))
    _, (X, y) = get_data()
    # TODO delete 2 lines below
    X = X[:10]
    y = y[:10]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")

    t = Trainer(X_train, y_train, **default_params)

    print(colored("## Training model ##", "blue"))
    t.train()

    print(colored("## Evaluating model ##", "blue"))
    t.evaluate(X_test, y_test)

    print(colored("## Saving model ##", "blue"))
    t.save_model()
