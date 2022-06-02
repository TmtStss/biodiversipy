import warnings

from termcolor import colored

from biodiversipy.data import get_data
from biodiversipy.trainer import Trainer

default_params = {
    'local': True,
}

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    print("############   Loading Data   ############")
    df, X_train, y_train = get_data()

    print(f"shape: {X_train.shape}")
    print(f"size: {X_train.memory_usage().sum() / 1e6} Mb")

    t = Trainer(X_train, y_train, **default_params)

    print(colored("############  Training model   ############", "red"))
    t.train()

    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()

    print(colored("############   Saving model    ############", "green"))
    t.save_model()
