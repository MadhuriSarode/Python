import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import LabelEncoder

# Reading the Data from CSV file
data = pd.read_csv('spam.csv',encoding='latin-1')

# Keeping only the neccessary columns
data = data[['v1','v2']]

# Pre processing data
data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

# Tokenization of Data, Converting to sequences
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)
X = pad_sequences(X)

# Model Configuration
embed_dim = 128
lstm_out = 196

# Method to create the model where required features is added and the model is compiled
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

# Applying Label Encoding on Target column used to transform non-numerical labels to numerical labels
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

# Model creation & Evaluation
batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 1, batch_size=32, verbose = 2)

# Loading the saved model and evaluating
score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=32)
print("Loss Score: ", (score))
print("Accuracy: %.2f%%" % (acc*100))
print(model.metrics_names)