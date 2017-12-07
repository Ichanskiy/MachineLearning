from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


df = pd.read_csv('OnlineNewsPopularity-02.csv', sep=',')
df = df.sample(frac=1).reset_index(drop=True)

Y = df.loc[450:495][' shares']
X = df[[' n_tokens_content', ' num_videos',' num_keywords',' average_token_length']].loc[450:495]

Y1 = df.loc[495:540][' shares']
X1 = df[[' n_tokens_content', ' num_videos',' num_keywords',' average_token_length']].loc[495:540]

trainX = X.values
trainY = Y.values
testX = X1.values
testY = Y1.values

tree = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski',
                            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                            weights='uniform')
print(tree)
tree = tree.fit(trainX, trainY)

predictY = tree.predict(testX)
print('оцінка точності', accuracy_score(testY, predictY))
print('показникик класифікації \n', metrics.classification_report(trainY, predictY, digits = 6))

fix, ax = plt.subplots()
ax.plot(list(range(46)), testY, label="актуальне")
ax.plot(list(range(46)), predictY, label="передбачення")
legend = ax.legend(loc="upper right")
legend.get_frame().set_facecolor('#FDF5E6')
plt.show()





