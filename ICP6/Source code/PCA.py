import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# You can add the parameter data_home to wherever to where you want to download your data
dataset = pd.read_csv('CC.csv')
x = dataset.iloc[:, 1:17]
y = dataset.iloc[:, -1]
# Eliminating the null values
x = x.select_dtypes(include=[np.number]).interpolate().dropna()


scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x)
# Apply transform to both the training set and the test set.
x_scaler = scaler.transform(x)

# Performing Principle Component Analysis (PCA)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, dataset[['TENURE']]], axis=1)
print(finaldf)


# Bonus: KMeans on PCA
# Performing K-Means clustering on the PCA data
nclusters = 3
km = KMeans(n_clusters=nclusters)
km.fit(x_pca)               # using x_pca data here instead of x

# Evaluation of the clusters accuracy
y_cluster_KMeans = km.predict(x_pca)
score = metrics.silhouette_score(x_pca, y_cluster_KMeans, metric='euclidean', sample_size=42)
print('Silhoutee Score of the Clusters using PCA is ', score)

# Elbow point computation to determine optimum number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Plotting the elbow point on graph
plt.plot(range(1, 11), wcss)
plt.title('The elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()