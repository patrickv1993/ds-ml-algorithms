
from pathlib import Path
import os
import csv
import json

_DATA_FOLDER = "data"

_DATA_CONVERSION_MAP = {
    "str": str,
    "int": int,
    "float": float,
}


def get_project_root():
    return str(Path().absolute().parent)


def read_csv(file_name: str, field_convert_map: dict = {},):
    file_path = os.path.join(get_project_root(), _DATA_FOLDER, file_name)
    with open(file_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        field_names = csv_reader.fieldnames
        csv_dict = {}
        n = 0
        for row in csv_reader:
            n += 1
            for field in field_names:
                val = row.get(field, "")

                if field in field_convert_map.keys():
                    val = _DATA_CONVERSION_MAP[field_convert_map[field]](val)

                csv_dict[field] = csv_dict.get(field, []) + [val]

    return csv_dict, n


def write_csv(file_name, csv_dict, csv_index, fieldnames,):
    with open(file_name, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index in csv_index:
            writer.writerow(csv_dict[index])

    return


def write_json(file_name, json_dict,):
    with open(file_name, "w") as outfile:
        json.dump(json_dict, outfile)


def dict_keys_to_list(d: dict,):
    return list(d.keys())


def dict_values_to_list(d: dict,):
    return list(d.values())


def list_subtraction(list1: list, list2: list,):
    return [l1 for l1 in list1 if l1 not in list2]


def list_addition(list1: list, list2: list):
    return list1 + [l2 for l2 in list2 if l2 not in list1]


def lists_to_dict(list1: list, list2: list,):
    return dict(zip(list1, list2))


def enumerate_list(list1: list,):
    return {i: v for i, v in enumerate(list1)}

