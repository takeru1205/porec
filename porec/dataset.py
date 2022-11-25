import pandas as Data
from pandas import DataFrame as DataSet


def read_csv(file_path: str) -> DataSet:
    return DataSet(Data.read_csv(file_path))
