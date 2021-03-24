# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Reading data
data = pd.read_csv('breastcancer.csv')

# Converting non-numerical data into numerical by replacing them
data['diagnosis'] = data['diagnosis'].replace('M', 0)
data['diagnosis'] = data['diagnosis'].replace('B', 1)
data = data.values


# Split the data set into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data[:, 2:32], data[:, 1], test_size=0.25,
                                                    random_state=87)

# Creating network model for breast cancer diagnosis
np.random.seed(155)
Cancer = Sequential()
# Provide input and neurons for first hidden dense layer
Cancer.add(Dense(15, input_dim=30, activation='relu'))
Cancer.add(Dense(1, activation='sigmoid'))

# Fitting the neural network model on the training data set
Cancer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
Cancer.fit(x_train, y_train, epochs=100, verbose=0, initial_epoch=0)

# Display the neural network identified
print('The summary of the neural network is', Cancer.summary())
score = Cancer.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('The metrics names', Cancer.metrics_names)
