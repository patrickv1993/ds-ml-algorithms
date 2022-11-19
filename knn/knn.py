
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


def fetch_from_distance_matrix(distance_dict: dict, i: int, j: int,):
    if i > j:
        return distance_dict[(i, j)]
    if i == j:
        return 0
    if i < j:
        return distance_dict[(j, i)]


def init_rank_dict(rank_dict, d, i,):
    rank_dict[1] = (i, d)
    min_dist = d
    max_dist = d
    return rank_dict, min_dist, max_dist


def shift_rank_dict(rank_dict, d, i, k,):
    if len(rank_dict) == k:
        rank_dict.pop(k)

    rank_dict = {k + 1: v for k, v in rank_dict.items()}
    rank_dict[1] = (i, d)
    min_dist = d
    max_dist = rank_dict[len(rank_dict)][1]
    return rank_dict, min_dist, max_dist


def split_rank_dict(rank_dict, d, i, k,):
    if len(rank_dict) == k:
        rank_dict.pop(k)

    r = min([len(rank_dict) + 1, k])
    is_current_max = True

    while r > 1 and is_current_max:
        old_d = rank_dict[r - 1][1]

        if old_d < d:
            is_current_max = False
        else:
            old_neighbor = rank_dict.pop(r - 1)
            rank_dict[r] = old_neighbor
            r -= 1

    rank_dict[r] = (i, d)
    min_dist = rank_dict[1][1]
    max_dist = rank_dict[len(rank_dict)][1]
    return rank_dict, min_dist, max_dist


def append_rank_dict(rank_dict, d, i, ):
    r = len(rank_dict) + 1
    rank_dict[r] = (i, d)
    min_dist = rank_dict[1][1]
    max_dist = rank_dict[r][1]
    return rank_dict, min_dist, max_dist


def find_k_nearest_neighbors(distance_dict, k, self_index, neighbor_index, ):
    nn_dict = {}
    for j in self_index.values():
        i_list = [i for i in neighbor_index.values() if i != j]
        rank_dict = {}
        for i in i_list:
            ij_distance = fetch_from_distance_matrix(distance_dict, i, j)
            if len(rank_dict) == 0:
                rank_dict, min_dist, max_dist = init_rank_dict(rank_dict, ij_distance, i,)
            elif ij_distance < min_dist:
                rank_dict, min_dist, max_dist = shift_rank_dict(rank_dict, ij_distance, i, k,)
            elif ij_distance < max_dist:
                rank_dict, min_dist, max_dist = split_rank_dict(rank_dict, ij_distance, i, k,)
            elif len(rank_dict) < k:
                rank_dict, min_dist, max_dist = append_rank_dict(rank_dict, ij_distance, i,)
            else:
                pass
        nn_dict[j] = rank_dict
    return nn_dict


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


    nn_dict = find_k_nearest_neighbors(distance_dict, k, test_index, train_index)

    return df_dict


if __name__ == "__main__":
    knn()
