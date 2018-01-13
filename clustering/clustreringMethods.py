import csv
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, AgglomerativeClustering, MiniBatchKMeans
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

countClusters = 3
a = 20 + 113
b = 12 * 20
c = 95 + 57
d = 5 * 11
e = 174 + 12
g = 9 * 14

print(a)
print(b)
print(c)
print(d)
print(e)
print(g)

row1 = []
row2 = []
row3 = []
row4 = []
row5 = []
row6 = []

with open('dim256.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        row1.append(row[a])
        row2.append(row[b])
        row3.append(row[c])
        row4.append(row[d])
        row5.append(row[e])
        row6.append(row[g])

myarray1 = np.asarray(row1)
myarray2 = np.asarray(row2)
myarray3 = np.asarray(row3)
myarray4 = np.asarray(row4)
myarray5 = np.asarray(row5)
myarray6 = np.asarray(row6)

X = np.array([myarray1, myarray4, myarray6])
X = X.transpose((1, 0))

kmeans = KMeans(n_clusters=countClusters, random_state=0, algorithm='elkan')
labels = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50,alpha=1, cmap='viridis')
plt.show()


sc = MiniBatchKMeans(countClusters, random_state=0)
labels = sc.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50,alpha=1, cmap='viridis')
plt.show()

sc = AgglomerativeClustering(n_clusters=countClusters, affinity='cosine', linkage='complete')
labels = sc.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50,alpha=1, cmap='viridis')
plt.show()


X = X[:100]
Z = hierarchy.linkage(X)
dn = hierarchy.dendrogram(Z)
plt.show()