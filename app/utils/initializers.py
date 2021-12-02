import pandas as pd
import os
import numpy as np
import joblib

from pandas import DataFrame
from enum import IntEnum
from typing import List
from glob import glob
from sklearn.preprocessing import MinMaxScaler

from app.models.options import Option

from typing import List
    
def get_files(option: IntEnum, is_training: bool = False) -> List[str]: 
    """get_files retrieves all the file paths related 
    to particular folder options. 
    1 = Building 1
    2 = Building 2 
    3 = Building 3
    4 = Training folder

    Args:
        option (IntEnum): an IntEnum that identifies each option 

    Returns:
        List[str]: a list of file paths. 
    """
    train_files = set(glob("datasets/building1/train/*.feather"))
    trajectory = "known" if is_training else "unknown"

    if option == Option.TRAIN: 
        return train_files
    else: 
        feather_files = set(glob(f"datasets/building{int(option)}/{trajectory}/*.feather"))
        feather_files = feather_files - train_files
        return feather_files

def create_data(files: List[str]) -> DataFrame:
    """create_data creates the training data by 
    taking all the feather files available, converting them 
    to data frames and then concatenating them together into 
    a single dataframe which is then converted into a CSV file. 

    Args:
        files (List[str]): a list of files

    Returns:
        DataFrame: a pandas dataframe
    """
    dfs = []

    for file in files: 
        df = pd.read_feather(file)
        mag_data = df[["iphoneMagX", "iphoneMagY", "iphoneMagZ"]].values.tolist()
        result = apply_minmax_scaling(mag_data)
        df2 = pd.DataFrame(result, columns=["iphoneMagX", "iphoneMagY", "iphoneMagZ"])
        df["iphoneMagX"] = df2["iphoneMagX"]
        df["iphoneMagY"] = df2["iphoneMagY"]
        df["iphoneMagZ"] = df2["iphoneMagZ"]

        dfs.append(df)
    
    df2 = pd.concat(dfs, ignore_index=True)
    return df2 

def apply_minmax_scaling(mag_data: List[List[float]]) -> List[List[float]]:
    """apply_minmax_scaling receives a matrix with magnetometer data
    which is then applied a minmax scaler from sklearn to normalize the data. 
    This is done to simulate Ellipsoid Fitting. A scaler is also saved 
    to reapply it later on with test data.

    Args:
        mag_data (List[List[float]]): nx3 matrix containing X, Y, Z coordinates
        with magnetometer data.

    Returns:
        List[List[float]]: nx3 matrix containing rescaled data. 
    """
    scaler = None
    file_path = "saves/scaler/minmax.save"
    results = None

    if os.path.exists(file_path):
        scaler = joblib.load(file_path)
        results = scaler.transform(mag_data)
        return results
    else: 
        scaler = MinMaxScaler()
        scaler.fit(mag_data)
        results = scaler.transform(mag_data)
        joblib.dump(scaler, file_path)
        return results


def get_latest_checkpoint(model: str, option: int) -> str:
    checkpoints_path = f"saves/{model}/checkpoints_{model}_building{option}"
    latest_checkpoint = ""


    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
        return latest_checkpoint
    
    checkpoints = glob(f"{checkpoints_path}/*.hdf5")

    if checkpoints:
        latest_checkpoint = checkpoints[-1]
    
    return latest_checkpoint


def initialize_data(option: IntEnum) -> DataFrame:
    """initialize_data will return a DataFrame with all the 
    data present for one building.  
    """

    files = get_files(option)
    df = create_data(files=files)

    return df


    