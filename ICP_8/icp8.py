# Import libraries
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

"""
Part 1
"""
# Read in csv as a dataframe
dataset = pd.read_csv("diabetes.csv", header=None).values
# print(dataset)

# Split the data into test and train portions
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(30, activation='relu')) # hidden layer
my_first_nn.add(Dense(4, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))

# Break up the output for readability
print('\n\n\n')

"""
Part 2
"""
# Read in csv as a dataframe
dataset = pd.read_csv("BreastCancer.csv", header=None).values

dataset = dataset[1:]

# Split the data to test data and target on the diagnosis
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:32], dataset[:,1],
                                                    test_size=0.25, random_state=87)

# Dummify the diagnosis data to binary
for idx,e in enumerate(Y_train):
    if e == 'B':
        e = 0
    else:
        e = 1
    Y_train[idx]=e

for idx,e in enumerate(Y_test):
    if e == 'B':
        e = 0
    else:
        e = 1
    Y_test[idx]=e

np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(35, activation='relu')) # hidden layer
my_first_nn.add(Dense(2, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))
