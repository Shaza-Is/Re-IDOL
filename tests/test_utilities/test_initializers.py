import pytest
import os 
import numpy as np
import pandas as pd

from glob import glob
from typing import List, Tuple

from app.utils.initializers import get_files, create_csv, create_data


@pytest.mark.skipif(os.path.isdir("datasets") == False, reason="Dataset not downloaded. Download dataset to run this test.")
@pytest.mark.parametrize("option", [1, 2, 3, 4])
def test_get_files(option: int) -> None:
    train_path = "datasets/building1/train/*.feather"
    building_path = f"datasets/building{option}/*/*.feather"

    file_path =  train_path if option == 4 else building_path 

    expected = len(set(glob(file_path)) - set(glob(train_path))) if option == 1 else len(glob(file_path))
    result = len(get_files(option))

    assert expected == result

@pytest.mark.skipif(os.path.isdir("datasets") == False, reason="Dataset not downloaded. Download dataset to run this test.")
def test_create_csv() -> None:
    data = np.random.rand(3, 3)
    df = pd.DataFrame(data=data, index = ["row1", "row2", "row3"], columns = ["column1", "column2", "column2"])
    file_path = "datasets/csvs/dummy.csv"
    create_csv(df, file_path)

    assert os.path.isfile(file_path)

    os.remove(file_path)

@pytest.mark.skipif(os.path.isdir("datasets") == False, reason="Dataset not downloaded. Download dataset to run this test.")
def test_create_data(generic_feathers: Tuple[List[str], int, int]) -> None:
    files, n_files, col_rows = generic_feathers
    df = create_data(files)

    size = n_files * col_rows

    assert len(df.index) == size




