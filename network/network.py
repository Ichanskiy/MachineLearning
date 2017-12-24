import pandas as pd
from matplotlib import pylab as pl
from pybrain3 import TanhLayer, SoftmaxLayer
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.utilities import percentError
import matplotlib.pyplot as plt


dataSet = pd.read_csv('winequality-red.csv', ';')

X = dataSet.as_matrix(columns=['citric acid', 'free sulfur dioxide', 'alcohol'])
X1 = dataSet.as_matrix(columns=['citric acid', 'free sulfur dioxide', 'alcohol'])
Y = dataSet.as_matrix(columns=['quality'])
Y1 = dataSet.as_matrix(columns=['quality'])

trainX = X[1:100]
trainY = Y[1:100]
testX = X1[100:199]
testY = Y1[100:199]
ds = SupervisedDataSet(trainX, trainY)
dsTest = SupervisedDataSet(testX, testY)


net = buildNetwork(3, 3, 1, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)
print(trainer.trainUntilConvergence(verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10))

p = net.activateOnDataset(dsTest)
fix, ax = plt.subplots()
ax.plot(list(range(99)), p, "C1o", label="спрогнозоване")
ax.plot(list(range(99)), testY, "ko", fillstyle="none", label="актуальне")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=4, mode="expand", borderaxespad=1.)
plt.show()
