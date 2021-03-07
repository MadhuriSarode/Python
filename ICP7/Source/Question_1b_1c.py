# Set the tfidf vectorizer parameter to use bigram and see how the accuracy changes
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

# Extracting data
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# Custom stop words to remove from the input data while forming the vector
my_stop_words = text.ENGLISH_STOP_WORDS.union(["book"])

tfidf_Vect = TfidfVectorizer()
tfidf_Vect1 = TfidfVectorizer(ngram_range=(1, 2), stop_words=my_stop_words)  # Here unigrams and bigrams are extracted from the input
tfidf_Vect2 = TfidfVectorizer(stop_words='english')                          # Removing stop words with no semantic value from the text

X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
X_train_tfidf1 = tfidf_Vect1.fit_transform(twenty_train.data)
X_train_tfidf2 = tfidf_Vect2.fit_transform(twenty_train.data)


clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('TfidfVectorizer score: ', score)


# Score of the model using unigrams and bigrams
clf1 = MultinomialNB()
clf1.fit(X_train_tfidf1, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf1 = tfidf_Vect1.transform(twenty_test.data)

predicted1 = clf1.predict(X_test_tfidf1)
score1 = metrics.accuracy_score(twenty_test.target, predicted1)
print("TfidfVectorizer score with ngram: " + str(score1))

# Score for X_train_tfidf using stop words as english
clf2 = MultinomialNB()
clf2.fit(X_train_tfidf2, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf2 = tfidf_Vect2.transform(twenty_test.data)

predicted2 = clf2.predict(X_test_tfidf2)

score3 = metrics.accuracy_score(twenty_test.target, predicted2)
print("TfidfVectorizer score with english stop words: " + str(score3))
