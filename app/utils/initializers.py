import pandas as pd
import os
import numpy as np

from pandas import DataFrame
from enum import IntEnum
from typing import List
from glob import glob


from app.models.options import Option
    
def get_files(option: IntEnum) -> List[str]: 
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

    if option == Option.TRAIN: 
        return train_files
    else: 
        feather_files = set(glob(f"datasets/building{int(option)}/*/*.feather"))
        feather_files = feather_files - train_files
        return feather_files

def create_csv(df: DataFrame, file_path: str) -> None:
    """create_csv creates a csv file from a DataFrame and 
    a filepath that is provided as an argument

    Args:
        df (DataFrame): a pandas DataFrame
        file_path (str): a string file path that points to some directory
    """
    df.to_csv(file_path)


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
        dfs.append(df)
    
    df2 = pd.concat(dfs, ignore_index=True)
    return df2


def create_test_data(files: List[str]) -> List[DataFrame]:
    """create_test_data takes a list of files and converts 
    them into DataFrame objects which are then stored into a list
    which is then returned to be used individually in order to 
    test trajectories. 

    Args:
        files (List[str]): a list of files

    Returns:
        List[DataFrame]: a list of DataFrames
    """
    dfs = []

    for file in files:
        df = pd.read_feather(file)
        dfs.append(df)

    return dfs


def generate_training_samples(matrix: np.ndarray, batch_size: int = 64):
    while True:

        acc = np.array([row[0:3].tolist() for row in matrix])
        gyro = np.array([row[3:6].tolist() for row in matrix])
        mag = np.array([row[6:9].tolist() for row in matrix])
        orient = np.array([row[9:].tolist() for row in matrix])

        xa_batch = np.zeros((batch_size,100,3))
        xg_batch = np.zeros((batch_size,100,3))
        xm_batch = np.zeros((batch_size,100,3))
        y_theta_batch = np.zeros((batch_size,4))
        y_sigma_batch = np.finfo(np.float32).eps* np.ones((batch_size,6)) ## To remove
        y_batch = np.concatenate((y_theta_batch, y_sigma_batch), axis=1)
        
        current_batch_number = 0
        for index in range(len(orient)):

            xa_batch[current_batch_number,:,:] = acc[index,:]
            xg_batch[current_batch_number,:,:] = gyro[index,:]
            xm_batch[current_batch_number,:,:] = mag[index,:]
            y_theta_batch[current_batch_number,:4] = orient[index,:]
            current_batch_number += 1
            if current_batch_number >= batch_size:
                current_batch_number = 0              
                yield([xa_batch,xg_batch, xm_batch],[y_batch])