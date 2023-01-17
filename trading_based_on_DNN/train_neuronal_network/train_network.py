import tensorflow as tf
from keras import models
from keras import layers
from keras.layers import Lambda
from keras.utils import to_categorical
from keras.models import model_from_json
import matplotlib.pyplot as plt


def get_price_list(path):
    data, data_prices = open(path, 'r'), []
    for line in data:
        v1, v2, v3 = line.rstrip().split(' | ')
        data_prices.append([int(float(v1)), float(v2), v3])
    return data_prices
                  

def create_input_state(i_timep, price_list):
    faktors = [1,1,1,1,1,1,1]  # Here you can define pattern where input for network should be calculated on: e.g.: [1,2,4,8,16,32,128]
    number = 10
    input_prices, sum_forward = [], 0
    for fak in faktors:
        for i in range(0, number, 1):
            try:
                input_prices.append(price_list[i_timep - sum_forward - i * fak][1])
                if i_timep-sum_forward-i*fak < 0:
                    print('Error')
            except:
                print('Error at timepoint: ', i_timep - sum_forward - i * fak)
        sum_forward += number*fak

    perc_input=[]
    for i_p in range(0, len(input_prices)-1, 1):
        before = input_prices[i_p+1]
        after = input_prices[i_p]
        perc = (after-before)/before*100
        perc_input += [round(perc,6)]
    return [perc_input]


def save_model(model):
    index_model = '__End__'
    model_json = model.to_json()
    with open("model" + str(index_model) + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model" + str(index_model) + ".h5")


def plot_training_history(history):
    acc = history.history['loss']
    val_acc = history.history['val_loss']
    Z_epochs = range(1, len(acc) + 1)
    plt.plot(Z_epochs, acc, 'b', label='Training loss', color='red')
    plt.plot(Z_epochs, val_acc, 'b', label='Validation loss', color='blue')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()


def get_data():
    # [1,0] = Long // [0,1] = Short
    sum_targ_perc, z_targ_perc = 0, 0
    list_path = ['28.20','29.20','30.20','31.20','32.20','33.20','34.20','35.20','36.20','37.20']
    hold_perc_threshold = 0.031
    steps_future = 4                           # Take the average of those steps in the future as trading target
    
    list_train_states,list_train_labels = [], []
    for week in range(0, 8, 1):  # first 80 %
        path = 'input_data/'+list_path[week]+'.txt'
        print('data: ', path)
        price_list = get_price_list(path)
        for i_time in range(70, len(price_list)-steps_future, 1):  # 70 = len(faktors) * number
            state = create_input_state(i_time, price_list)
            sum_dif_perc = 0
            for targ in range(0, steps_future, 1):
                dif_perc = (price_list[i_time+targ][1]-price_list[i_time][1])/price_list[i_time][1]*100
                sum_dif_perc += dif_perc
            target=sum_dif_perc/steps_future
            sum_targ_perc += abs(target)
            z_targ_perc += 1
                
            if target > hold_perc_threshold:  # Long
                list_train_labels.append([1, 0])
                list_train_states.append(state)
            elif target < -hold_perc_threshold:  # Short
                list_train_labels.append([0, 1])
                list_train_states.append(state)
    print('Number train_States  //  Number train_labels: ', len(list_train_states), '  //  ', len(list_train_labels))

    list_test_states, list_test_labels = [], []
    for week in range(8, 10, 1):  # last 20 %
        path = 'input_data/'+list_path[week]+'.txt'
        print('data: ', path)
        price_list = get_price_list(path)
        for i_time in range(70, len(price_list)-steps_future, 1):
            state = create_input_state(i_time, price_list)
            sum_dif_perc = 0
            for targ in range(0, steps_future, 1):
                dif_perc = (price_list[i_time+targ][1]-price_list[i_time][1])/price_list[i_time][1]*100
                sum_dif_perc += dif_perc
            target = sum_dif_perc/steps_future
            sum_targ_perc += abs(target)
            z_targ_perc += 1
            
            if target > hold_perc_threshold:  # Long
                list_test_labels.append([1, 0])
                list_test_states.append(state)
            elif target < -hold_perc_threshold:  # Short
                list_test_labels.append([0, 1])
                list_test_states.append(state)
    print('Number test_states)  //  Number test_labels: ', len(list_test_states), '  //  ', len(list_test_labels))
    print('Average target in percent: ', round(sum_targ_perc/z_targ_perc, 4))
    return list_train_states, list_train_labels, list_test_states, list_test_labels


train_states, train_labels, test_states, test_labels = get_data()

model = models.Sequential()
model.add(layers.LSTM(69, input_dim=69, return_sequences=True, activation='tanh'))
model.add(layers.LSTM(32, activation='tanh', return_sequences=True))
model.add(layers.LSTM(11, activation='tanh'))
model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])                                  
model.summary()

filepath = "saved-model-{epoch:02d}.hdf5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_best_only = False, save_weights_only = False, save_freq='epoch')

history = model.fit(train_states, train_labels, epochs=20, batch_size=8, shuffle=False, verbose=2, validation_data=(test_states, test_labels),
                    callbacks=[model_checkpoint])  # Alternative:  batch_size=32, validation_split=0.2, shuffle=True, verbose=2)
save_model(model)
plot_training_history(history)
