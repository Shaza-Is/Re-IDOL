import pytest
import os
import numpy as np

from app.services.preprocessor import PreProcessor
from tensorflow.keras import Input

@pytest.mark.skipif(os.path.isdir("datasets") == False, reason="Dataset not downloaded. Download dataset to run this test.")
def test_preprocessor_reshape_data(preprocessor: PreProcessor) -> None:
    pre = PreProcessor("datasets/csvs/train.csv", "datasets/csvs/building1.csv")

    X_train, y_train = pre.reshape_data()

    assert X_train.shape[1] == 9
    assert y_train.shape[1] == 4

