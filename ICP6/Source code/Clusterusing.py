import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

sns.set(style="white", color_codes=True)
import warnings

warnings.filterwarnings("ignore")

# Read input data from csv file
dataset = pd.read_csv('CC.csv')
# Assigning x with feature set and y with target feature
x = dataset.iloc[:, 1:17]
y = dataset.iloc[:, -1]
print(x.shape, y.shape)

# Check null values
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:25])
print('Null count of the columns = ', nulls)

# Question 1 a) Replacing the null values with mean value of the feature
meanMinPyt = dataset.loc[:, "MINIMUM_PAYMENTS"].mean()
print('Mean of Minimum Payments is ', meanMinPyt)
x = x.fillna(meanMinPyt)

# Grouping the tenure with it's values
print(dataset["TENURE"].value_counts())

# Cluster identification
sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "CREDIT_LIMIT", "BALANCE_FREQUENCY").add_legend()
sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "PURCHASES_FREQUENCY", "CREDIT_LIMIT").add_legend()
sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "PURCHASES_FREQUENCY", "CASH_ADVANCE_FREQUENCY").add_legend()
sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "MINIMUM_PAYMENTS", "CREDIT_LIMIT").add_legend()
sns.FacetGrid(dataset, hue="TENURE", size=4).map(plt.scatter, "BALANCE", "CREDIT_LIMIT").add_legend()
plt.show()

# Question 1 b)Preprocessing the data
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns=x.columns)

nclusters = 3  # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)

score = metrics.silhouette_score(x, y_cluster_kmeans)
print('Score = ', score)

# Question 1 c) elbow method to know the number of clusters
wcss = []
for i in range(2, 17):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 17), wcss)
plt.title('The elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------
# Question 2 : Using feature scaling and then applying KMeans on the scaled features.

# Finding the top 5 most correlated columns to the target variable
corr = dataset.corr()
print(corr['TENURE'].sort_values(ascending=False)[:6])

# Eliminating the null values
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()

# assigning data columns selected from correlation value
x = data.iloc[:, [2, 3, -4, -5, -6]]
y = data.iloc[:, -1]

#  elbow method to know the number of clusters
wcss = []
for i in range(2, 17):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 17), wcss)
plt.title('The elbow method - feature selection')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
