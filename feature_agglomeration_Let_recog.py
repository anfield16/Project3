

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics, random_projection
from sklearn.cluster import FeatureAgglomeration
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale


df = pd.read_csv("letter-recognition.csv", header=None) # !!!type is dataframe, not ndarray!!
df = np.array(df)  # type conversion needed to use slicing
# dfs = df[(df[:, 0] == 'A') | (df[:, 0] == 'B') | (df[:, 0] == 'C') | (df[:, 0] == 'D') |
#          (df[:, 0] == 'E') | (df[:, 0] == 'F') | (df[:, 0] == 'G') | (df[:, 0] == 'H') |
#          (df[:, 0] == 'I') | (df[:, 0] == 'J') | (df[:, 0] == 'K') | (df[:, 0] == 'L') |
#          (df[:, 0] == 'M') | (df[:, 0] == 'N') | (df[:, 0] == 'O')]
data = df[:, 1:]
data = scale(data)
labels = df[:, 0]
n_digits = len(np.unique(labels))

print (data.shape)
print (data.std())
std_transf = np.zeros(15)
x_axis = range(2,17)
for i in x_axis:
    transformer = FeatureAgglomeration(n_clusters=i)
    rp = transformer.fit_transform(data)
    print (rp.shape)
    std_transf[i-2] = rp.std()
plt.plot(x_axis, std_transf)
plt.ylabel('STD of data')
plt.xlabel('n_components')
plt.title("STD Vs Dimenions For Feature Agglomeration")
plt.show()


# ###############################################################################
# # Visualize the results on PCA-reduced data
#
# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
#
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
