import pandas as pd
import sklearn
import sklearn.linear_model as lm
import statsmodels.api as sm
import mpmath as math


# зчитуєм файл
dataSet = pd.read_csv('winequality-red.csv', ';')
dataSet.head()
# записуєм відповідні змінні
y = dataSet[['quality']].loc[:200]
x = dataSet[['citric acid', 'free sulfur dioxide', 'alcohol']].loc[:200]
# обчислюємо коефіцієнти регресії
smm = sm.OLS.from_formula("dataSet.quality ~ dataSet['citric acid'] + dataSet['free sulfur dioxide'] + dataSet.alcohol", data=dataSet)
result = smm.fit()
print(result.params.values)
b0 = result.params[0]
b1 = result.params[1]
b2 = result.params[2]
b3 = result.params[3]
after = b0 + b1*dataSet['citric acid'].loc[:200] + b2*dataSet['free sulfur dioxide'].loc[:200] + b3*dataSet['alcohol'].loc[:200]
print("R2 = ", abs(sklearn.metrics.r2_score(dataSet['quality'].loc[:200], after)))


def getSum(lists):
    sum = 0
    for i in lists:
        sum = sum + i
    return sum


def getDobutok(list1, list2):
    sum = 0
    for i in range(len(list1)):
        sum = sum + list1[i] + list2[i]
    return sum

def getDispersiya(list1, x2):
    sum = 0
    for i in list1:
        sum = sum + i*i
    return sum/len(list1) - x2*x2


def getError(x1, y1):
    y1_ = getSum(y1) / 200
    x1_ = getSum(x1) / 200
    x1y1 = getDobutok(x1, y1) / 200
    Dx1 = getDispersiya(x1, x1_)
    Dy1 = getDispersiya(y1, y1_)
    r_x1y1 = (x1y1 - y1_ * x1_) / (math.sqrt(Dx1) * math.sqrt(Dy1))
    error = abs(r_x1y1 * ((math.sqrt(200 - 2)) / (1 - r_x1y1 * r_x1y1)))
    print(error)

y1 = dataSet[['quality']].loc[:200].values.flatten()
x1 = dataSet[['citric acid']].loc[:200].values.flatten()
x2 = dataSet[['free sulfur dioxide']].loc[:200].values.flatten()
x3 = dataSet[['alcohol']].loc[:200].values.flatten()
getError(x1, y1)
getError(x2, y1)
getError(x3, y1)



