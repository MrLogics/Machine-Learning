import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM

df_train = pd.read_csv(r"C:\Users\wsfja\Documents\GitHub\Machine-Learning\Data\movie-review\train.tsv", sep = '\t')

replace_list = {r"i'm": 'i am',
		r"'re": ' are',
		r"let's": 'let us',
		r"'s": ' is',
		r"'ve": ' have',
		r"can't": 'can not',
		r"cannot": 'can not',
		r"shan't": 'shall not',
		r"n't": ' not',
		r"'d": ' would',
		r"'ll": ' will',
		r"'scuse": 'excuse',
		',': ' ,',
		'.': ' .',
		'!': ' !',
		'?': ' ?',
		'\s+': ' '}


def clean_text(text):
	text = text.lower()
	for s in replace_list:
		text = text.replace(s, replace_list[s])
	return text


X_train = df_train['Phrase'].apply(lambda p:clean_text(p))
phrase_len = X_train.apply(lambda p:len(p.split(' ')))
max_phrase_len = phrase_len.max()
plt.figure(figsize = (10,8))
plt.hist(phrase_len, alpha = 0.2, density = True)
plt.xlabel('phrase_len')
plt.ylabel('probability')
plt.show()

Y_train = df_train['Sentiment']
max_words = 8192

tokenizer = Tokenizer(num_words = max_words, filters = '*#$%&()+-/:;<=>@[\]^_`{|}~')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen = max_phrase_len)
Y_train = to_categorical(Y_train)

model = Sequential()
model.add(Embedding(input_dim = max_words, output_dim = 256, input_length = max_phrase_len))
model.add(Dropout(0.25))
model.add(LSTM(256))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.15))
model.add(Dense(5, activation = 'relu'))

model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

bs = 128
e = 4
H = model.fit(X_train, Y_train, validation_split = 0.15, epochs = e, batch_size = bs, verbose = 2)

N = range(1, e+1)
plt.clf()
plt.plot(N, H.history['loss'], label = 'trainLoss')
plt.plot(N, H.history['val_loss'], label = 'valLoss')
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.clf()
plt.plot(N, H.history['accuracy'], label = 'trainAccuracy')
plt.plot(N, H.history['val_accuracy'], label = 'valAccuracy')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
