import statistics
from typing import List, Tuple

import math
import numpy
from sklearn import datasets
from sklearn import preprocessing


def calculate_h_x(theta: Tuple[float], x: Tuple[float]) -> float:
    return 1.0 / (1.0 + math.exp(-(sum([a * b for (a, b) in zip(theta, x)]))))


def calculate_cost(theta: Tuple[float, ...], x_n: List[Tuple[float]], y: List[float], reg_lambda: float = 0.0) -> float:
    reg_result = reg_lambda * sum([theta_j ** 2 for theta_j in theta[1:]])
    return sum([(calculate_h_x(theta, x_n_i) - y_i) ** 2 + reg_result for (x_n_i, y_i) in zip(x_n, y)]) / (2.0 * len(y))


def calculate_partial_cost(theta: Tuple[float], x_n: List[Tuple[float]], y: List[float], j: int, reg_lambda: float = 0.0) -> float:
    reg_result = 0 if j == 0 else reg_lambda * theta[j]
    return sum([(calculate_h_x(theta, x_n_i) - y_i) * x_n_i[j] + reg_result for (x_n_i, y_i) in zip(x_n, y)]) / len(y)


def __add_x_0(x: Tuple[float, ...]) -> Tuple[float, ...]:
    return (1.0,) + x


def preprocess_x_n(x_n: List[Tuple[float, ...]], standard=True) -> List[Tuple[float, ...]]:
    """
    对于各组数据进行预处理（增加x_0特性，进行标准化）
    :param x_n: 数据集
    :param standard: 是否进行标准化
    :return: 预处理后的数据集
    """
    return [__add_x_0(tuple(x)) for x in (preprocessing.scale(numpy.array(x_n)).tolist() if standard else x_n)]


def preprocess_y(y):
    return preprocessing.scale(y)


# 迭代法求解
def regress(x_n: List[Tuple[float, ...]], y: List[float], *, tolerant: float = 1e-5, alpha: float = 0.3, regular_lambda: float = 0.0,
            standard=True) -> tuple:
    assert len(x_n) == len(y)
    assert len(x_n) != 0

    n = len(x_n[0]) + 1

    processed_x_n = preprocess_x_n(x_n, standard=standard)
    processed_y = y

    cur_theta = (0.0,) * n
    while True:
        before_cost = calculate_cost(cur_theta, processed_x_n, processed_y, reg_lambda=regular_lambda)

        cur_theta = tuple(
            [(cur_theta[j] - alpha * calculate_partial_cost(cur_theta, processed_x_n, processed_y, j, reg_lambda=regular_lambda))
             for j in range(0, n)])

        after_cost = calculate_cost(cur_theta, processed_x_n, processed_y, reg_lambda=regular_lambda)
        if abs(after_cost - before_cost) < tolerant:
            break

    return cur_theta


# X, Y = pandas_read_csv_data("classify.csv")
# X = X.tolist()
# Y = Y.tolist()
# train_data = X[:-2]
# train_answer = Y[:-2]
# test_data = X[-2:]
# test_answer = Y[-2:]

X, Y = datasets.load_breast_cancer(return_X_y=True)
X = X.tolist()
Y = Y.tolist()
train_data = X[:-85]
train_answer = Y[:-85]
test_data = X[-85:]
test_answer = Y[-85:]

final_theta = regress(train_data, train_answer)
print(final_theta)


def predict(x_n, test_x_n, test_y, theta, norm=True):
    x_n_t = numpy.array(x_n).transpose().tolist()
    feature_statistics = [(statistics.mean(feature), statistics.pstdev(feature)) for feature in x_n_t]
    l2_diff = []
    for (test_x, y) in zip(test_x_n, test_y):
        processed_test_x = [1.0] + [((x - f[0]) / f[1]) if norm else x for (x, f) in zip(test_x, feature_statistics)]
        h_x = calculate_h_x(theta, tuple(processed_test_x))
        h_x = 1 if h_x >= 0.5 else 0
        print("predict result is %f and diff is %f" % (h_x, abs(y - h_x)))
        l2_diff.append((y - h_x) ** 2)
    print("l2-norm: %f" % math.sqrt(sum(l2_diff) / len(l2_diff)))


predict(train_data, test_data, test_answer, final_theta)
