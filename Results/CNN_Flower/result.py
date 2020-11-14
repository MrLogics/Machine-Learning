Python 3.8.5 (tags/v3.8.5:580fbb0, Jul 20 2020, 15:57:54) [MSC v.1924 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
= RESTART: C:\Users\wsfja\Documents\GitHub\Machine-Learning\04. CNN\CNN_Flowers.py
Train data shape:  (693, 128, 128, 3)
Validation data shape:  (123, 128, 128, 3)
Test data shape:  (144, 128, 128, 3)
WARNING:tensorflow:From C:\Users\wsfja\Documents\GitHub\Machine-Learning\04. CNN\CNN_Flowers.py:102: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
Please use Model.fit, which supports generators.
Epoch 1/5
WARNING:tensorflow:From C:\Users\wsfja\AppData\Local\Programs\Python\Python38\lib\site-packages\tensorflow\python\training\tracking\tracking.py:111: Model.state_updates (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
WARNING:tensorflow:From C:\Users\wsfja\AppData\Local\Programs\Python\Python38\lib\site-packages\tensorflow\python\training\tracking\tracking.py:111: Layer.updates (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
This property should not be used in TensorFlow 2.0, as updates are applied automatically.
44/44 - 56s - loss: 2.3460 - accuracy: 0.1313 - val_loss: 107.2555 - val_accuracy: 0.2276
Epoch 2/5
44/44 - 59s - loss: 1.9124 - accuracy: 0.3001 - val_loss: 119.0669 - val_accuracy: 0.4309
Epoch 3/5
44/44 - 35s - loss: 1.7263 - accuracy: 0.3853 - val_loss: 96.7259 - val_accuracy: 0.4146
Epoch 4/5
44/44 - 34s - loss: 1.5614 - accuracy: 0.4343 - val_loss: 110.3720 - val_accuracy: 0.5122
Epoch 5/5
44/44 - 30s - loss: 1.5424 - accuracy: 0.4430 - val_loss: 128.0424 - val_accuracy: 0.5041
Model Evaluation
1/5 [=====>........................] - ETA: 0s - loss: 173.9251 - accuracy: 0.43752/5 [===========>..................] - ETA: 0s - loss: 175.6902 - accuracy: 0.35943/5 [=================>............] - ETA: 0s - loss: 158.0425 - accuracy: 0.40624/5 [=======================>......] - ETA: 0s - loss: 147.1002 - accuracy: 0.41415/5 [==============================] - ETA: 0s - loss: 137.3963 - accuracy: 0.42365/5 [==============================] - 1s 248ms/step - loss: 137.3963 - accuracy: 0.4236
[137.39630126953125, 0.4236111044883728]
Model Prediction
[ 4  8  8 10  8  4  0 10  1 10  4 10  7 10 10  3 10 10  2 10  4  9 10 10
 10 10 11  0 10  9 10  6  0  8  3  0  8  3  2  0 10  8  3  2  6  0 10 10
  4  0  0 11  6 10  9 10  3 11  6  9  8  4  9  9  9  6 10  4  8  8  3  0
  2 10  6  0  9  4 10  4  1  0  8  0 10  6 11  4  8 10  0 10  1  6  6  8
  9  3 10 11 10  2  0  0  2  8  6  5  6 10  0  8  8 10 11  9  1  6  0  2
  1 11  4  3  8 10 10  6  7  0  4 10  1  4  1  6  1  0  6  9  9  0 10  6]
[ 4  2  4  7  6  4  9  3  1  5  4 10  7  5 10 11  5 10  0  6  4  2  3 10
 10 10 11  4  1  9  1  7  2  2  3  7  7  3  2  9  5  3 10  8  8  2 10  5
  4  2  6 11  2  1  7 10  6  1  7  9  8  7  7  2  9  7 10  4  8  9  5  9
  2  1  6  0  2  4 10  4  3  9  8  9  1  2  1  4  0  3  9 10  3  6  6  8
 11  8  1  3  5  9  7  0  8  9  6  5  6 10  0  8  2 10 11  7  1  6  9  9
 11  3  4  3  0  1  5  6  7  0  4  1 11  4 11  6  1  0  8  0  0  2 10  0]
>>> 