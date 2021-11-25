from uuid import UUID

from app.utils.initializers import get_files, create_csv, create_data


def do_preprocessing_work(pid: UUID, file_path: str, option: int):
    """do_work is a thread safe function to allow
    multiple processes preprocess data for neural network 
    training.

    Args:
        file_path (str): A filepath that contains the
        correct path to create the preprocessed data file

        option (int): An option to select which file to preprocess,
        1 = train
        2 = building1
        3 = building2
        4 = building3  
    """

    files = get_files(option)
    df = create_data(files=files)

    print(f"Process with pid {pid} creating file: {file_path}")

    create_csv(df, file_path)


def do_training_work():
    pass