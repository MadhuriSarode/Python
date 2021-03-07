from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# Considering only few categories for the model fit.
categories = ['alt.atheism', 'soc.religion.christian']
# Extracting training data
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)

# SVM model fit for the training data
tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = SVC()
clf.fit(X_train_tfidf, twenty_train.target)
# Testing the model with the test data
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)
# Comparing predicted and actual test output to estimate the accuracy of the model
score = metrics.accuracy_score(twenty_test.target, predicted)
print("Score with SVC: " + str(score))


