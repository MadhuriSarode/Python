# Question2 : Add embedding layer to the model, did you experience any improvement?

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten

# Reading data from csv file into dataframe
df = pd.read_csv('imdb_master.csv', encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values

# tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
# getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# The input dimension, to count the number of elements, all done against one input axis.
input_dim = np.prod(X_train.shape[1:])

# Number of features
print(input_dim)



# Sequential model implementation
model = Sequential()
model.add(layers.Dense(300, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
# History object holding the trained model fit for the training data
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


# Adding embedding layer

# Pre processing for Embedding Layer
pureSentences = df['review'].values
max_review_len = max([len(s.split()) for s in pureSentences])
print('max_review_len', max_review_len)
vocab_size = len(tokenizer.word_index) + 1
sentencesPre = tokenizer.texts_to_sequences(pureSentences)
padded_docs = pad_sequences(sentencesPre, maxlen=max_review_len)

X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)
print('vocab_size', vocab_size)

# Model implementation adding Embedding Layer
model_embedded = Sequential()
model_embedded.add(Embedding(vocab_size, 50, input_length=max_review_len))
model_embedded.add(Flatten())
model_embedded.add(layers.Dense(300, activation='relu', input_dim=max_review_len))
model_embedded.add(layers.Dense(3, activation='softmax'))
model_embedded.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history1 = model_embedded.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

# Model Evaluation using test dataset
[test_loss1, test_acc1] = model_embedded.evaluate(X_test, y_test)
print("Evaluation result on Test Data after embedding : Loss = {}, accuracy = {}".format(test_loss1, test_acc1))


# Loss and Accuracy Curve after adding Embedding Layer
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuray', 'val_accuracy','loss','val_loss'], loc='upper left')
plt.show()

# Bonus question 2 : Predict over one sample of data and check what will be the prediction for that
pred = model_embedded.predict_classes(X_test[[2], :])
print("Actual Prediction", y_test[1], "Predicted Prediction", pred)

