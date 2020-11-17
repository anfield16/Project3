

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale


df = pd.read_csv("letter-recognition.csv", header=None) # !!!type is dataframe, not ndarray!!
df = np.array(df)  # type conversion needed to use slicing
# df=df[:500,:]
samples = df.shape[0]

partition = np.array(range(1, 104, 5)) / 100.0

# a=np.random.randint(0, samples, int(samples * partition[0]))
# print a
# print type(a)
# dft=df[a,:]
# print dft.shape
# dd

# array to store results
homo = np.zeros(len(partition))
for i in range(len(partition)):
    df_part = df[np.random.randint(0, samples, int(samples * partition[i])), :]
    data = df_part[:, 1:]
    data = scale(data)
    labels = df_part[:, 0]
    n_digits = len(np.unique(labels))
    reduced_data = FastICA(n_components=6, max_iter=200).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    homo[i] = metrics.homogeneity_score(labels, kmeans.labels_)

print(homo)

plt.plot(partition, homo, marker='*', linestyle='--', color='g')
plt.ylabel('Homogeneity')
plt.xlabel('Percentage of sample used')
plt.title("Homogeneity Vs Percent of sample , ICA + KMeans")
plt.show()
