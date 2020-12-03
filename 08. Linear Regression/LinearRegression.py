import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabetes.data)
# print(diabetes.DESCR)

# Only one feature to plot the graph
# diabetes_X = diabetes.data[:, np.newaxis, 2]

# For all features
diabetes_X = diabetes.data

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_predict = model.predict(diabetes_X_test)

print("Mean Squared error is: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# Output for One feature
# Mean Squared error is:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698

# To plot it
# plt.scatter(diabetes_X_test, diabetes_Y_test)
# plt.plot(diabetes_X_test, diabetes_Y_predict)
# plt.show()

# Output for all features
# Mean Squared error is:  1826.5364191345425
# Weights:  [  -1.16924976 -237.18461486  518.30606657  309.04865826 -763.14121622
#   458.90999325   80.62441437  174.32183366  721.49712065   79.19307944]
# Intercept:  153.05827988224112

# It cannot be plotted.


