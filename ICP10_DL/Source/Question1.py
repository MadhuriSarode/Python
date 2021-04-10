
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# Question 1 : there are three mistake which stop the code to get run successfully;
# find those mistakes and explain why they need to be corrected to be able to get the code run.

df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values
#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Number of features
# print(input_dim)

# Mistake 1 : The input dimension value is missing and knowing the input training elements size, we get
# 2000 number words, the input_dim should be 2000.
model = Sequential()
model.add(layers.Dense(300,input_dim=2000, activation='relu'))

# Mistake 2 : We have a multi-class classification problem, the activation function should be Softmax
# which is used for multi-classification in the Logistic Regression model.Here the outputs are mutually exclusive.
# Whereas Sigmoid is used for binary classification in the Logistic Regression model.
model.add(layers.Dense(3, activation='Softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)
