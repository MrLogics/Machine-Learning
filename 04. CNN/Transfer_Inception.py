#import for Model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Dropout, BatchNormalization
from keras.layers.core import Dense
from keras.layers.core import Flatten

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Input, GlobalAveragePooling2D
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy

#import for training
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import cv2
from imutils import paths
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics

def top_2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true = y_true, y_pred = y_pred, k=2)

h, w, d = 227, 227, 3
n_classes = 12
epochs = 10
bsize = 8
data = []
labels = []

input_tensor = Input(shape = (h, w, d))

base_model = InceptionV3(input_tensor = input_tensor, weights = 'imagenet', include_top = False)
X = base_model.output
X = GlobalAveragePooling2D()(X)
X = Dense(200, activation = 'relu')(X)
predictions = Dense(n_classes, activation = 'softmax')(X)

lab_dict = {'bluebell':0, 'buttercup':1, 'crocus':2, 'daffodil':3, 'daisy':4, 'dandelion':5, 'iris':6, 'lilyvalley':7, 'pansy':8, 'snowdrop':9, 'sunflower':10, 'tulip':11}  

imagepaths = sorted(list(paths.list_images("..\Data\Flowers")))

random.seed(0)
random.shuffle(imagepaths)

for imgpath in imagepaths:
    image = cv2.imread(imgpath)
    image = cv2.resize(image,(h, w))
    image = np.array(image)
    image = preprocess_input(image)
    data.append(image)
    lab = imgpath.split(os.path.sep)[-2]
    labels.append(lab_dict[lab])
    
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

aug = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                                          shear_range=0.2, zoom_range=0.2, horizontal_flip='True', fill_mode="nearest") 

model = Model(input = base_model.input, output = predictions)

for layer in model.layers[:-40]:
    layers.trainable = True
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy', top_2])

p=3
es = EarlyStopping(monitor = "val_accuracy", patience = p)
ck = ModelCheckpoint(filepath = "..\Results\Transfer_Inception\flowers_best.model", monitor = "val_accuracy", save_best_only = 'True')
callbacks = [es, ck]

H = model.fit_generator(aug.flow(trainX, trainY, batch_size = bsize), validation_data = (ValX, ValY), epochs = epochs, verbose = 2, callbacks = callbacks)


plt.style.use("ggplot")
plt.figure(figsize = [8, 6])
N = len(H.history["loss"])
plt.plot(np.arange(0, N), H.history["loss"], label = "train-loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("..\Results\Transfer_Inception\Loss.png")

plt.style.use("ggplot")
plt.figure(figsize = [8, 6])
N = len(H.history["loss"])
plt.plot(np.arange(0, N), H.history["accuracy"], label = "train-acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label = "val-acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("..\Results\Transfer_Inception\Accuracy.png")

print("Model Evaluation")
print(model.evaluate(testX, testY))

print("Model Prediction")
predY = model.predict(testX)
predY = np.argmax(predY, axis = 1)
testY = np.argmax(testY, axis = 1)
print(predY)
print(testY)


