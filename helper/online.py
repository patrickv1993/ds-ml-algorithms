from math import sqrt
from math import dist

def init_summary_stat_dict(data: list, n: int,):
    payload = {
        "raw_data": data,
        "data": [],
        "n_rem": n,
        "n": 0,
        "sum": 0,
        "sq_sum": 0,
    }
    return payload


def get_stats_from_dict(summary_stat_dict: dict, stat_col: str,):
    summary_stat_dict[stat_col].pop("n_rem")
    N = summary_stat_dict[stat_col].pop("n")
    x_bar = summary_stat_dict[stat_col].pop("sum") / N
    var_x = (1 / N) * summary_stat_dict[stat_col].pop("sq_sum") - x_bar ** 2
    sd_x = sqrt(var_x)
    summary_stat_dict[stat_col]["x_bar"] = x_bar
    summary_stat_dict[stat_col]["sd_x"] = sd_x
    return summary_stat_dict


def summary_stat_standardization(
        df_dict: dict,
        n: int,
        stat_cols: list,):
    summary_stat_dict = {
        col: init_summary_stat_dict(df_dict[col], n,)
        for col in stat_cols
    }

    i = 0
    while i < n:
        for col in stat_cols:
            raw_data = summary_stat_dict[col]["raw_data"].pop(0)
            summary_stat_dict[col]["sum"] += raw_data
            summary_stat_dict[col]["sq_sum"] += raw_data ** 2
            summary_stat_dict[col]["n_rem"] -= 1
            summary_stat_dict[col]["n"] += 1
            summary_stat_dict[col]["data"] += [raw_data]

        i += 1

    std_df_dict = {}
    for col in stat_cols:
        summary_stat_dict[col].pop("raw_data")
        data = summary_stat_dict[col].pop("data")
        summary_stat_dict = get_stats_from_dict(summary_stat_dict, col)
        std_data = [(d - summary_stat_dict[col]["x_bar"]) / summary_stat_dict[col]["sd_x"] for d in data]
        df_dict[col] = data
        std_df_dict[col] = std_data

    return summary_stat_dict, std_df_dict


def df_dict_to_distance(
        std_df_dict: dict,
        n: int,):
    print(1)

    j = 0
    k = 0

    distance_dict = {}

    while j < n:
        i = j + 1
        left_vector = [v.pop(0) for v in std_df_dict.values()]
        while i < n:
            right_vector = [v[i - j - 1] for v in std_df_dict.values()]
            distance_dict[(i, j)] = dist(left_vector, right_vector)
            k += 1
            i += 1
        j += 1

    return distance_dict
