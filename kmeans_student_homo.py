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

df = pd.read_csv("student-etoh-por.csv", sep=';', header=0)  # !!!type is dataframe, not ndarray!!
# print 'shape of data: ', df.shape

# preprocessing, change string to int
df = df.apply(preprocessing.LabelEncoder().fit_transform)
df = np.array(df)  # type conversion needed to use slicing

all_column = np.arange(25)  # select feature for prediction
all_column = np.append(all_column, [28, 29, 30, 31])
X = df[:, all_column]
y = df[:, 26]
cluster_num = len(np.unique(y))
sample_size = 300

print("n_categories: %d, \t n_samples %d, \t n_features %d"
      % (cluster_num, X.shape[0], 30))


print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=cluster_num, n_init=10),
              name="k-means++", data=X)

bench_k_means(KMeans(init='random', n_clusters=cluster_num, n_init=10),
              name="random", data=X)

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

