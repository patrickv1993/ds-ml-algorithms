
from pathlib import Path
import os
import csv

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
