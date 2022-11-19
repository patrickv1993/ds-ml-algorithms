
from helper import utils
from helper.constants import REPO_SEED, TRAIN_TEST_SPLIT
import random

_DEFAULT_DATA = "iris.csv"

_DEFAULT_DATA_SCHEMA = {
    "Sepal.Length": "float",
    "Sepal.Width": "float",
    "Petal.Length": "float",
    "Petal.Width": "float",
    "Species": "str",
}

_DEFAULT_K = 5


def configure_run(seed: int = REPO_SEED,
                  k: int = _DEFAULT_K,
                  train_test_split: float = TRAIN_TEST_SPLIT,):
    random.seed(seed)
    return k, train_test_split


def load_df(file_name: str, schema: dict = {},):
    df_dict = utils.read_csv(
        file_name=file_name,
        field_convert_map=schema,
    )
    return df_dict


def knn():
    k, train_test_split = configure_run()
    df_dict = load_df(
        file_name=_DEFAULT_DATA,
        schema=_DEFAULT_DATA_SCHEMA,
    )
    return df_dict


if __name__ == "__main__":
    knn()
