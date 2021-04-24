import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


# Reading the Data from CSV file
data = pd.read_csv('Sentiment.csv')
# Keeping only the neccessary columns for sentiment analysis
data = data[['text', 'sentiment']]

# Pre processing data
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

# Tokenization of Data, Converting to sequences
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)

# Model Configuration
embed_dim = 128
lstm_out = 196


# Method to create the model where required features is added and the model is compiled
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Applying Label Encoding on Target column used to transform non-numerical labels to numerical labels
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model creation & Evaluation
batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs = 1, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=batch_size)

# Saving the model as h5 format for easier loading from next time
from keras.models import load_model
model.save("sentiment_analysis.h5")

# Loading the saved model and evaluating
loaded_model = load_model("sentiment_analysis.h5")
loss, accuracy = loaded_model.evaluate(X_test, Y_test)
print("The Loss is ",loss)
print("The Accuracy is ",accuracy)

# Processing the input text
input_text = "A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump"
tokenizer.fit_on_texts(input_text)

#vectorizing the tweet by the pre-fitted tokenizer instance
tweet = tokenizer.texts_to_sequences(input_text)

tweet = pad_sequences(tweet, maxlen=28, dtype='int32', value=0)

# Sentiment Prediction of the text from the saved h5 model
result = loaded_model.predict(tweet,batch_size=1,verbose = 2)[0]
print(result)
if(np.argmax(result) == 0):
    print("negative")
elif (np.argmax(result) == 1):
    print("positive")
else:
    print("neutral")

