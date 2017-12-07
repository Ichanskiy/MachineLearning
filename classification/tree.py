from sklearn import tree, metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

dataSet = pd.read_csv('OnlineNewsPopularity-02.csv', sep=',')
dataSet = dataSet.sample(frac=1).reset_index(drop=True)

Y = dataSet.loc[450:495][' shares']
X = dataSet[[' n_tokens_content', ' num_videos', ' num_keywords', ' average_token_length']].loc[450:495]

Y1 = dataSet.loc[495:540][' shares']
X1 = dataSet[[' n_tokens_content', ' num_videos', ' num_keywords', ' average_token_length']].loc[495:540]

trainX = X.values
trainY = Y.values
testX = X1.values
testY = Y1.values

tree = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10, max_features=None, max_leaf_nodes=5,
                                   min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None,
                                   splitter='random')
print(tree)
tree = tree.fit(trainX, trainY)

# побудова дерева
# export_graphviz(tree, feature_names=[' n_tokens_content', ' num_videos', ' num_keywords', ' average_token_length'], filled=False)


predictY = tree.predict(testX)
# print('оцінка точності', accuracy_score(testY, predictY))
print('показникик класифікації \n', metrics.classification_report(trainY, predictY, digits = 6))
fix, ax = plt.subplots()
ax.plot(list(range(46)), testY, label="актуальне")
ax.plot(list(range(46)), predictY, label="передбачення")
legend = ax.legend(loc="upper right")
legend.get_frame().set_facecolor('#FDF5E6')
plt.show()





