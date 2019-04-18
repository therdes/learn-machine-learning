import os
from typing import Tuple, List

import numpy
import pandas


def read_data_from_file(file_path) -> Tuple[List[tuple], List[float]]:
    data = []
    answer = []
    if os.path.exists(file_path) and os.path.splitext(file_path)[1] == ".csv":
        with(open(file_path, "r")) as f:
            for line in f.read().splitlines():
                line_data = line.split(",")
                data.append(tuple([float(item) for item in line_data[:-1]]))
                answer.append(float(line_data[-1]))
    return data, answer


def pandas_read_csv_data(file_path, header=None, answer_index=-1) -> Tuple[numpy.ndarray, numpy.ndarray]:
    csv_dataset: pandas.DataFrame = pandas.read_csv(file_path, header=header, dtype=float)
    train_data: numpy.ndarray = csv_dataset.iloc[:, :answer_index].values
    train_answer: numpy.ndarray = csv_dataset.iloc[:, answer_index].values
    return train_data, train_answer
