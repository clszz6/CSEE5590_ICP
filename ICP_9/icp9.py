from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

(train_images,train_labels), (test_images, test_labels) = mnist.load_data()

#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /=255.0
test_data /=255.0
#change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#creating network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
[train_loss, train_acc] = model.evaluate(train_data, train_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
print("Evaluation result on Trained Data : Loss = {}, accuracy = {}".format(train_loss, train_acc))


"""
Part 1
"""
graph = plot.subplot()
graph.plot(history.history['loss'], label='trained loss')
graph.plot(history.history['val_loss'], label='validated loss')
graph.set(xlabel='epoch', ylabel='loss', title='trained loss VS validated loss')
plot.show()

graph.clear()
graph = plot.subplot()
graph.plot(history.history['acc'], label='trained accuracy')
graph.plot(history.history['val_acc'], label='validated accuracy')
graph.set(xlabel='epoch', ylabel='acc', title='trained acc VS validated acc')
plot.show()
# After seeing the graphs it appears to NOT be overfitted


"""
Part 2
"""
single_prediction = model.predict(train_data[0].reshape(1, 784))
number_prediction = np.argmax(single_prediction)

plt.imshow(train_images[0,:,:],cmap='gray')
plt.title('Ground Truth : {}, Prediction: {}'.format(train_labels[0], number_prediction))

# Plots the prediction on the graph
plt.show()


"""
Part 3
"""
model = Sequential()
model.add(Dense(512, activation='tanh', input_shape=(dimData,)))
model.add(Dense(512, activation='tanh'))
model.add(Dense(512, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result with 3 hidden layers and tanh activation: Loss = {}, accuracy = {}".format(test_loss, test_acc))
# Results with the new model is that the accuracy seems to be the same but the loss is better
