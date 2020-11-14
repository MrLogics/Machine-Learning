from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.utils import to_categorical

from matplotlib import pyplot as plt
import random

import cv2
from imutils import paths
import numpy as np
import os
from sklearn.model_selection import train_test_split


def createmodel(h, w, d, c):
    model = Sequential()
    
    #first layer
    model.add(Conv2D(50, (5,5), padding = "same", input_shape = (h, w, d)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    
    #Second layer
    model.add(Conv2D(30, (3,3), padding = "same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation("relu"))
    
    model.add(Dense(c))
    model.add(Activation("softmax"))
    
    return model

h, w, d = 28, 28, 3
n_classes = 10
epochs = 5
bsize = 8
data = []
labels = []

imagepaths = sorted(list(paths.list_images("..\Data\DigitDataset")))

random.seed(0)
random.shuffle(imagepaths)

for imgpath in imagepaths:
    image = cv2.imread(imgpath)
    image = cv2.resize(image,(h, w))
    image = np.array(image)
    data.append(image)
    lab = imgpath.split(os.path.sep)[-2]
    labels.append(lab)
    
data = np.array(data, np.float64)
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.15, random_state = 0)
(trainX, ValX, trainY, ValY) = train_test_split(trainX, trainY, test_size = 0.15, random_state = 0)

print("Train data shape: ", trainX.shape)
print("Validation data shape: ", ValX.shape)
print("Test data shape: ", testX.shape)

trainY = to_categorical(trainY, num_classes = n_classes)
ValY = to_categorical(ValY, num_classes = n_classes)
testY = to_categorical(testY, num_classes = n_classes)

model = createmodel(h, w, d, n_classes)
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])

H = model.fit(trainX, trainY, batch_size = bsize, validation_data = (ValX, ValY), epochs = epochs, verbose = 2)

model.save("..\Results\CNN_MNIST\digits.model")

plt.style.use("ggplot")
plt.figure(figsize = [8, 6])
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label = "train-loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("..\Results\CNN_MNIST\Loss.png")

plt.style.use("ggplot")
plt.figure(figsize = [8, 6])
N = epochs
plt.plot(np.arange(0, N), H.history["accuracy"], label = "train-acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label = "val-acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("..\Results\CNN_MNIST\Accuracy.png")

print("Model Evaluation")
print(model.evaluate(testX, testY))

print("Model Prediction")
predY = model.predict(testX)
predY = np.argmax(predY, axis = 1)
testY = np.argmax(testY, axis = 1)
print(predY)
print(testY)


