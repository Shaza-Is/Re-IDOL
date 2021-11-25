import pandas as pd
import os

from pandas import DataFrame
from typing import List
from glob import glob

from app.models.options import Option
    
def get_files(option: int) -> List[str]: 
    train_files = set(glob("datasets/building1/train/*.feather"))

    if option == Option.TRAIN: 
        return train_files
    else: 
        feather_files = set(glob(f"datasets/building{int(option)}/*/*.feather"))
        feather_files = feather_files - train_files
        return feather_files

def create_csv(df: DataFrame, file_path: str) -> None:
    df.to_csv(file_path)


def create_data(files: List[str]) -> DataFrame:
    dfs = []

    for file in files: 
        df = pd.read_feather(file)
        dfs.append(df)
    
    df2 = pd.concat(dfs, ignore_index=True)
    return df2
