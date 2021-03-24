import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Read the data from csv file
dataset = pd.read_csv("diabetes.csv", header=None).values

# Split the data into test and training dataset
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8],
                                                    test_size=0.25, random_state=87)

# create sequential model
np.random.seed(155)
my_first_nn = Sequential()

# Add hidden layers
my_first_nn.add(Dense(20, input_dim=8, activation='relu', name='layer1'))  # hidden layer
my_first_nn.add(Dense(15, activation='relu'))  # output layer
my_first_nn.add(Dense(10, activation='relu'))  # output layer
my_first_nn.add(Dense(5, activation='relu'))  # output layer
my_first_nn.add(Dense(1, activation='sigmoid'))  # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)

# will give you the display labels for the scalar outputs(to find out what each of those values corresponds to)
print('The metrics names', my_first_nn.metrics_names)
score = my_first_nn.evaluate(X_test, Y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

