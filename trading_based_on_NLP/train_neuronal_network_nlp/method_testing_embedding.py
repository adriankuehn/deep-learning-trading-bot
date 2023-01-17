from keras import preprocessing
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


"""  Check whether Embedding Layer works  """

max_features = 10000 
maxlen = 20 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) 
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen) 
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train[0:10]: ', x_train[0:10])
print('y_train[0:10]: ', y_train[0:10])


model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen)) 
model.add(Flatten()) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)
