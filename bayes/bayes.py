import csv
import math
import matplotlib.pyplot as plt


# загрузка даних з файлу
def readCsv(file):
    lines = csv.reader(open(file, "rt"))
    dataFrame = list(lines)
    for i in range(len(dataFrame)):
        dataFrame[i] = [float(k) for k in dataFrame[i]]
    return dataFrame


# відділення даних по класам
# example {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
def divideByClass(dataFrame):
    divided = {}
    for i in range(len(dataFrame)):
        v = dataFrame[i]
        if (v[-1] not in divided):
            divided[v[-1]] = []
        divided[v[-1]].append(v)
    return divided


# середнє
def getAVG(listNumbers):
    return sum(listNumbers) / float(len(listNumbers))


# стандартне відхилення
def standardDeviation(numbers):
    avg = getAVG(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers))
    return math.sqrt(variance)


# сумування
# example: dataset = [[1,20,0], [2,21,1], [3,22,0]] ---> [(2.0, 1.0), (21.0, 1.0)]
def sumDataSet(dataFrame):
    summaries = [(getAVG(attribute), standardDeviation(attribute)) for attribute in zip(*dataFrame)]
    del summaries[-1]
    return summaries


# сумування атрибітів по класам
def summarizeByClass(dataset):
    separated = divideByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = sumDataSet(instances)
    return summaries


# розрахунок ймовірнсоті атрибута що належить класу
def getProbability(x, avg, sd):
    if sd == 0:
        sd = 1e-9
    exp = math.exp(-(math.pow(x - avg, 2) / (2 * math.pow(sd, 2))))
    return (1 / (math.sqrt(2 * math.pi) * sd)) * exp


# розрахунок ймовірності приналежнсоті класу
def getClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= getProbability(x, mean, stdev)
    return probabilities


# розрахунок очікування
def getPredict(summaries, inputVector):
    probabilities = getClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = getPredict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# розрахунок цінки точності
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


trainFileName = 'train_repl.csv'
testFileName = 'test_repl.csv'
trainingDataSet = readCsv(trainFileName)
testDataSet = readCsv(testFileName)
print('вибрано навчальних = {0} і тестових = {1} рядків даних'.format(len(trainingDataSet), len(testDataSet)))
# вчимо модель
summaries = summarizeByClass(trainingDataSet)
# тестуємо на тестовій вибірці і отримаємо прогноз
predictions = getPredictions(summaries, testDataSet)
accuracy = getAccuracy(testDataSet, predictions)
print('Точність: ',  round(accuracy, 2))


actual = list(testDataSet[i][-1] for i in range(len(testDataSet)))
fix, ax = plt.subplots()
ax.plot(list(range(44)), predictions, "C1o", label="спрогнозоване")
ax.plot(list(range(44)), actual, "ko", fillstyle="none", label="актуальне")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, ncol=4, mode="expand", borderaxespad=1.)
plt.show()