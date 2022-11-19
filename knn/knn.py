
from helper import utils

_DEFAULT_DATA = "iris.csv"

_DEFAULT_DATA_SCHEMA = {
    "Sepal.Length": "float",
    "Sepal.Width": "float",
    "Petal.Length": "float",
    "Petal.Width": "float",
    "Species": "str",
}


def load_df(file_name: str, schema: dict = {},):
    df_dict = utils.read_csv(
        file_name=file_name,
        field_convert_map=schema,
    )
    return df_dict


def knn():
    df_dict = load_df(
        file_name=_DEFAULT_DATA,
        schema=_DEFAULT_DATA_SCHEMA,
    )
    return df_dict


if __name__ == "__main__":
    knn()
