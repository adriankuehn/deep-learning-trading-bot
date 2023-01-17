from keras import models
from keras import layers


def save_model(model, epoch):
    model_json = model.to_json()
    with open("model_final_"+str(epoch)+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_final_"+str(epoch)+".h5")

    
def create_model():
    model = models.Sequential()
    model.add(layers.LSTM(69, input_dim=69, return_sequences=True, activation='tanh'))
    model.add(layers.LSTM(32, activation='tanh', return_sequences=True))
    model.add(layers.LSTM(11, activation='tanh'))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])                                  
    model.summary()
    return model


epoch = "15"
print('Load model of epoch ', str(epoch))
model = create_model()
model.load_weights('saved-model-'+str(epoch)+'.hdf5')
save_model(model, epoch)
