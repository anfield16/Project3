#############################################################################
## code for dataset: letter-recognition
## precision is

from time import time
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    return (metrics.homogeneity_score(y, estimator.labels_))

df = pd.read_csv("student-etoh-por.csv", sep=';', header=0)  # !!!type is dataframe, not ndarray!!
# print 'shape of data: ', df.shape

# preprocessing, change string to int
df = df.apply(preprocessing.LabelEncoder().fit_transform)
df = np.array(df)  # type conversion needed to use slicing

num_feature = np.array([2,4,6,8,10,12,14,16,18,20,22,24])
max_homo = np.zeros(len(num_feature))
for feature in range(len(num_feature)):
    all_column = np.arange(25)  # select feature for prediction
    all_column = np.append(all_column, [28, 29, 30, 31])
    homo = np.zeros(20)
    for i in range(20):
        selected_col = np.random.choice(all_column, 8, replace = False)
        X = df[:, selected_col]
        y = df[:, 26]
        cluster_num = len(np.unique(y))
        sample_size = 300

        homo[i] = bench_k_means(KMeans(init='k-means++', n_clusters=cluster_num, n_init=10),
                      name="k-means++", data=X)
    max_homo[feature] = max(homo)
print max_homo
# plt.figure()
# # plt.scatter(X, y, c="darkorange", label="data")
# plt.scatter(dimensions, homo, c="orange")
# plt.plot(dimensions, homo, color="blue", label="accuracy_score", linewidth=2)
# # plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("Dimensions of kmeans)")
# plt.ylabel("accuracy_score")
# plt.title("learning_rate value on GradientBoost prediction")
# plt.legend()
# plt.show()

