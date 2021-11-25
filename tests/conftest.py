import pytest
import numpy as np
import os
import pyarrow as pa
import pyarrow.feather as feather
import pandas as pd

from app.services.preprocessor import PreProcessor
from typing import List 


@pytest.fixture
def preprocessor() -> PreProcessor:
    return PreProcessor("datasets/csvs/train.csv", "datasets/csvs/building1.csv")

@pytest.fixture
def generic_csv() -> str:
    arr = np.random.rand(20, 20)
    path = "datasets/csvs/dummy.csv"
    np.savetxt(path, arr)

    yield path

    os.remove(path)

@pytest.fixture
def generic_feathers() -> List[str]: 
    base_path = "datasets/gen"
    files = []
    n_files = 30
    col_rows = 20

    rows = [f"row{x}" for x in range(0, col_rows)]
    columns = [f"column{x}" for x in range(0, col_rows)]

    for number in range(0, n_files):
        arr = np.random.rand(col_rows , col_rows)
        df = pd.DataFrame(arr, index = rows, columns = columns)
        file_path = f"{base_path}/gen_{number}.feather"
        files.append(file_path)
        feather.write_feather(df, file_path)

    yield (files, n_files, col_rows)

    for file in files: 
        os.remove(file)



