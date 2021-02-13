# Implement Naïve Bayes method using scikit-learn library

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

# Read the input csv file
glass = pd.read_csv('glass.csv')

# All the columns other than type are loaded in X, and 'type' column in Y.
X = glass.drop('Type', axis=1)
Y = glass['Type']

# Split dataset into training set and test set with 20% of the data for testing and 80% for training
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.4)

# Correlation function between X and Y columns
corrmat = glass.corr()
f, ax = plt.subplots(figsize=(9, 8))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu",
            linewidths=0.1)  # Heat map showing the correlation of variables among themselves
plt.show()

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets, fitting the Naive Bytes model to training data
model.fit(X_train, Y_train)

# Predict the response for test dataset using the model built
Y_predicted_output = model.predict(X_test)

# Model Accuracy is calculated by comparing predicted output by the model and the actual output
print("accuracy score:", metrics.accuracy_score(Y_test, Y_predicted_output) * 100)
print(metrics.classification_report(Y_test, Y_predicted_output))
