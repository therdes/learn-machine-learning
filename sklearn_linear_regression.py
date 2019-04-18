import math
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor

# train_data, train_answer = pandas_read_csv_data("house_price.csv")
# test_data = [[1750, 3, 2, 35]]

# X, Y = pandas_read_csv_data("ConcreteTable.csv")
# X = X.tolist()
# Y = Y.tolist()
# train_data = X[:-50]
# train_answer = Y[:-50]
# test_data = X[-50:]
# test_answer = Y[-50:]

# X, Y = pandas_read_csv_data("classify.csv")
# X = X.tolist()
# Y = Y.tolist()
# train_data = X[:-2]
# train_answer = Y[:-2]
# test_data = X[-2:]
# test_answer = Y[-2:]

X, Y = datasets.load_iris(return_X_y=True)
train_data = X[:-5]
train_answer = Y[:-5]
test_data = X[-5:]
test_answer = Y[-5:]

if False:
    least_squares_regressor = LinearRegression().fit(np.array(train_data), np.array(train_answer))
    print("theta : %s" % least_squares_regressor.coef_)
    l2_diff = []
    for (h_x, y) in zip(least_squares_regressor.predict(np.array(test_data)), test_answer):
        print("predict %f with %f, diff %f" % (h_x, y, abs(h_x - y)))
        l2_diff.append((h_x - y) ** 2)
    print("l2-norm: %f" % (math.sqrt(sum(l2_diff) / len(l2_diff))))

if False:
    sgd_regressor = SGDRegressor(tol=1e-5).fit(np.array(train_data), np.array(train_answer))
    for (h_x, y) in zip(sgd_regressor.predict(np.array(test_data)), test_answer):
        print("predict %f with %f, diff %f" % (h_x, y, abs(h_x - y)))

if True:
    logistic_regression = LogisticRegression(solver='lbfgs', multi_class='auto').fit(train_data, train_answer)
    for (h_x, y) in zip(logistic_regression.predict(test_data), test_answer):
        print("predict %f with %f, diff %f" % (h_x, y, abs(h_x - y)))
