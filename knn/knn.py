
from helper import utils
from helper.constants import REPO_SEED, TRAIN_TEST_SPLIT
from helper.online import summary_stat_standardization, df_dict_to_distance
from helper.sampling import sample_and_split
import random

_DEFAULT_DATA = "iris.csv"

_DEFAULT_DATA_SCHEMA = {
    "Sepal.Length": "float",
    "Sepal.Width": "float",
    "Petal.Length": "float",
    "Petal.Width": "float",
    "Species": "str",
}

_DEFAULT_X_COLUMNS = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
_DEFAULT_Y_COLUMN = "Species"

_DEFAULT_K = 5


def configure_run(seed: int = REPO_SEED,
                  k: int = _DEFAULT_K,
                  train_test_split: float = TRAIN_TEST_SPLIT,):
    random.seed(seed)
    return k, train_test_split


def load_df(file_name: str, schema: dict = {},):
    df_dict, n = utils.read_csv(
        file_name=file_name,
        field_convert_map=schema,
    )
    return df_dict, n


def knn():
    k, train_test_split = configure_run()
    df_dict, n = load_df(
        file_name=_DEFAULT_DATA,
        schema=_DEFAULT_DATA_SCHEMA,
    )
    summary_stat_dict, std_df_dict = summary_stat_standardization(
        df_dict=df_dict,
        n=n,
        stat_cols=_DEFAULT_X_COLUMNS,
    )

    distance_dict = df_dict_to_distance(std_df_dict, n,)

    df_index = list(range(n))
    test_index, train_index = sample_and_split(
        sequence=df_index,
        p=train_test_split,
    )

    return df_dict


if __name__ == "__main__":
    knn()
