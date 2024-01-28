import logging
import os
from functools import partial

import numpy as np
import pandas as pd

ASSETS_ROOT = 'assets/'
LIB_OCCURRENCE_THRESHOLD = 50
TARGET_FIELD = "is_virus"
MODEL_STORE_FILE = "model.bin"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("basic_logger")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def fix_lib_list(libs_list: str) -> list[str]:
    lib_list = libs_list.split(",")
    lib_list = list(map(lambda string: string.split("/")[-1], lib_list))  # ignore path
    return list(set(lib_list))


def get_source_files() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train = pd.read_csv(os.path.join(ASSETS_ROOT, "train.tsv"), sep="\t")
    validation = pd.read_csv(os.path.join(ASSETS_ROOT, "val.tsv"), sep="\t")
    test = pd.read_csv(os.path.join(ASSETS_ROOT, "test.tsv"), sep="\t")
    return train, validation, test


def calculate_classes_balance(train: pd.DataFrame) -> (int, int):
    with_virus_cnt = len(train[train[TARGET_FIELD] == 1])
    without_virus_cnt = len(train[train[TARGET_FIELD] == 0])
    return with_virus_cnt, without_virus_cnt


def create_libs_set(dataframe: pd.DataFrame, lib_occurrence_threshold: int) -> set:
    """
    get unique libs found in dataframe, which number of occurrences >= threshold
    """
    list_libs_2d = list(map(fix_lib_list, dataframe["libs"]))
    list_libs_1d = np.concatenate(list_libs_2d)
    libs_count = {}
    for lib in list_libs_1d:
        if lib not in libs_count:
            libs_count[lib] = 0
        libs_count[lib] += 1
    set_libs = set()
    for lib, count in libs_count.items():
        if count >= lib_occurrence_threshold:
            set_libs.add(lib)
    return set_libs


def calculate_presence_of_libs(row: str, indexes: dict[str, int], lib_number: int) -> list[bool]:
    current_libs = fix_lib_list(row)
    libs_presence = [False] * lib_number
    for curr_lib in current_libs:
        if curr_lib in indexes:
            libs_presence[indexes[curr_lib]] = True
    return libs_presence


def fix_dataframe(df: pd.DataFrame, set_libs: set) -> pd.DataFrame:
    """
    create dataframe like one-hot encoded:
    +--------------+--------------+---------+------------------------+------------+-----------+
    | is_virus     | kernel32.dll | libc.so | ... (len(set)) total)  | user32.dll | gdi32.dll |
    +--------------+--------------+---------+------------------------+------------+-----------+
    | 1            | 1            | 0       | 1                      | 0          | 1         |
    | 1            | 0            | 1       | 1                      | 1          | 0         |
    | ... (N more) |              |         |                        |            |           |
    | 0            | 1            | 0       | 1                      | 1          | 0         |
    +--------------+--------------+---------+------------------------+------------+-----------+
    """
    libs_index = {}
    for index, unique_lib in enumerate(list(set_libs)):
        libs_index[unique_lib] = index

    columns = df["libs"].map(partial(calculate_presence_of_libs, indexes=libs_index, lib_number=len(set_libs)))
    columns = pd.DataFrame(columns.tolist(), columns=list(set_libs))  # slow operation
    df = df.join(columns)
    df.drop(["libs"], inplace=True, axis=True)
    return df
