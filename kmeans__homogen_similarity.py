

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

df = pd.read_csv("letter-recognition.csv", header=None) # !!!type is dataframe, not ndarray!!
df = np.array(df)  # type conversion needed to use slicing

# pairs that looks similar, thus may not be clustered well
similar_pair = np.array([['I', 'J'], ['U', 'V'],  ['D', 'O'], ['O', 'Q']])
#print(similar_pair[1,1])

diff_pair = np.array([['S', 'T'], ['I', 'W'], ['J', 'Z'], ['L', 'M']])

# array to store results
homo = np.zeros(len(similar_pair)+len(similar_pair))

for i in range(0, len(similar_pair)):
    df_similar = df[(df[:, 0] == similar_pair[i, 0]) | (df[:, 0] == similar_pair[i, 1])]
    df_similar_data = df_similar[:, 1:]
    df_similar_label = df_similar[:, 0]
    kmeans_similar = KMeans(n_clusters=2).fit(df_similar_data)
    homo[i] = metrics.homogeneity_score(df_similar_label, kmeans_similar.labels_)

for i in range(0, len(diff_pair)):
    df_similar = df[(df[:, 0] == diff_pair[i, 0]) | (df[:, 0] == diff_pair[i, 1])]
    df_similar_data = df_similar[:, 1:]
    df_similar_label = df_similar[:, 0]
    kmeans_similar = KMeans(n_clusters=2).fit(df_similar_data)
    homo[len(similar_pair) + i] = metrics.homogeneity_score(df_similar_label, kmeans_similar.labels_)
print(homo)

# plt.plot(homo[0:len(similar_pair)])


z = range(1, 9)
n = ['I-J','U-V','D-O','O-Q','S-T','I-W','J-Z','L-M']

fig, ax = plt.subplots()
ax.scatter(z, homo, c='g')

for i, txt in enumerate(n):
    ax.annotate(txt, (z[i],homo[i]))
plt.ylabel('homogeneity')
plt.xlabel('pair of letters')
plt.title("Pair of non similar letters gives higher homogeneity")
plt.show()





# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
# pca = PCA(n_components=n_digits).fit(data) #n_components=26 must be between 0 and n_features=16 with svd_solver='full'
# bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#               name="PCA-based",
#               data=data)
# print(79 * '_')

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
