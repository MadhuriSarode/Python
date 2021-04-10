from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten
from sklearn.datasets import fetch_20newsgroups

# Importing the dataset after Downloading 20news dataset.
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)

# Feature Extraction
sentences = newsgroups_train.data
# Target extraction
y = newsgroups_train.target

# Converting text sentences into tokens.
tokenizer = Tokenizer(num_words=2000)
max_review_len = max([len(s.split()) for s in sentences])
vocab_size = len(tokenizer.word_index) + 1
sentencesPre = tokenizer.texts_to_sequences(sentences)
padded_docs = pad_sequences(sentencesPre, maxlen=max_review_len)
# Splitting the data into test and training sets
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)


# Sequential model implementation with embedding layer
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_review_len))
model.add(Flatten())
model.add(layers.Dense(300, activation='relu', input_dim=max_review_len))
model.add(layers.Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# History object holding the trained model fit for the training data
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)


# Model Evaluation using test dataset
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Predicting the Value for test sample
pred = model.predict_classes(X_test[[2], :])
print("Actual Prediction", y_test[1], "Predicted Prediction", pred)


# Bonus question1 :  Plotting the accuracy and loss in a graph using history object.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuray', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')
plt.show()
