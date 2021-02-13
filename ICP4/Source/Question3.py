import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn import model_selection
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

glass = pd.read_csv("glass.csv")

# Preprocessing data
X = glass.drop('Type', axis=1)
Y = glass['Type']

# Split dataset into training set and test set with 60% of the data for testing and 40% for training
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.4)

# SVM model is trained and it is fitted for the training data
svc = SVC()
svc.fit(X_train, Y_train)
Y_predicted_output = svc.predict(X_test)
# Model Accuracy is calculated by comparing predicted output by the model and the actual output
print("accuracy score:", metrics.accuracy_score(Y_test, Y_predicted_output) * 100)
print(metrics.classification_report(Y_test, Y_predicted_output))
