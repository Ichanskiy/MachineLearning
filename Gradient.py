import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataSet = pd.read_csv('winequality-red.csv', ';')
property = dataSet.iloc[:200]
X = np.matrix(property[['citric acid', 'free sulfur dioxide', 'alcohol']])
y = np.matrix(property['quality'])

def getDifference(theta, x):
    return theta.T * x.T

def gradient(x, y, a, iteration):
    m, n = np.shape(x)
    beta = [[1.84], [0.73], [0.00006], [0.3]]
    for i in range(iteration):
        beta = beta - a / m * x.T * (x * beta - y.T)
    return beta

n = X.shape[0]
first = np.matrix(np.ones(n))
x = np.hstack((first.T, X))
alpha = 0.000001
iteration = 500000
beta = gradient(x, y, alpha, iteration)
print(beta)

dataSetNew = dataSet.iloc[200:300]
x1 = np.hstack((np.matrix(np.ones(100)).T, np.matrix(dataSetNew[['citric acid', 'free sulfur dioxide', 'alcohol']])))
yOld = getDifference(beta, x1)
error = np.matrix(yOld) - np.matrix(dataSetNew['quality'])
error = list(map(abs, error.tolist()[0]))
x = list()
for i in range(100):
    x.append(i)
plt.plot(x, error)
plt.show()
