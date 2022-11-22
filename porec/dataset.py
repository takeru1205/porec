import pandas as pd
from pandas import DataFrame as DataSet


def read_csv(file_path: str) -> DataSet:
    return DataSet(pd.read_csv(file_path))
