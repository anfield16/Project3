

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import scale


df = pd.read_csv("letter-recognition.csv", header=None) # !!!type is dataframe, not ndarray!!
df = np.array(df)  # type conversion needed to use slicing
#df = df[:200, :]
data = df[:, 1:]
data = scale(data)
labels = df[:, 0]
n_digits = len(np.unique(labels))

dimensions = range(2, 18)

# array to store results
homo = np.zeros(len(dimensions))

for dim in dimensions:
    if dim <= 16:
        reduced_data = FeatureAgglomeration(n_clusters=dim).fit_transform(data)
    else:
        reduced_data = data
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    homo[dim-2] = metrics.homogeneity_score(labels, kmeans.labels_)

print(homo)


plt.plot(dimensions[:-1], homo[:-1], marker='*', linestyle='--', color='r')
plt.plot(dimensions[-1], homo[-1], marker='D', color='b')
plt.ylabel('Homogeneity')
plt.xlabel('Dimensions After FeatureAgglomeration reduction')
plt.title("Homogeneity Vs Dimension with FeatureAgglomeration + Kmeans")
plt.show()