import keras
from keras import layers
import matplotlib.pyplot as plt

    
def plot_train_history(history):
    acc = history.history['loss']
    val_acc = history.history['val_loss']
    z_epochs = range(1, len(acc) + 1)
    plt.plot(z_epochs, acc, 'b', label='Training loss', color='red')
    plt.plot(z_epochs, val_acc, 'b', label='Validation loss', color='blue')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()


def normalize_main(states):
    def normalize(stat):
        list_stat = []
        for st in range(0, len(stat), 1):
            Norm = round(0.5 + (stat[st] / 2), 8)
            if Norm>1 or Norm<0:
                print('Fatal Error lop 1: ', Norm)
                lop
            list_stat.append(Norm)
        return list_stat
    
    list_norm = []
    for state in states: list_norm.append(normalize(state[0]))
    return list_norm


def save_model(model, au_en):
    model_json = model.to_json()
    if au_en == 1: name = "autoencoder"
    elif au_en == 2: name = "encoder"
    with open(name+".json", "w") as json_file: json_file.write(model_json)
    model.save_weights(name+".h5")


def get_price_list(path):
    data, data_prices = open(path, 'r'), []
    for line in data:
        v1, v2, v3 = line.rstrip().split(' | ')
        data_prices.append([int(float(v1)), float(v2), v3])
    return data_prices
                  

def create_input(timep_indeces_c, price_list_c):
    factors = [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1]
    number, input_state, sum_s = 10, [], 0
    for fak in factors:
        for i in range(0, number, 1):
            try:
                input_state.append(price_list_c[timep_indeces_c - sum_s - i * fak][1])
                if timep_indeces_c-sum_s-i*fak < 0:
                    print('BULLSHITTTTTTTTTTTTTT')
                    lop
            except:
                print('Position: ', timep_indeces_c - sum_s - i * fak)
                lop
        sum_s += number*fak
    perc_input = []
    for i_p in range(0, len(input_state)-1, 1):
        before = input_state[i_p+1]
        after = input_state[i_p]
        perc = (after-before)/before*100
        perc_input.append(round(perc,6))
    return [[perc_input]]


def calculate_train_data():
    # [1,0] = Long // [0,1] = Short
    sum_targ_perc, z_targ_perc = 0, 0
    paths=['32.20','33.20','34.20','35.20','36.20','37.20']
    hold_perc_threshold = 0.065
    target = 40
    list_train_states, list_train_labels = [], []

    for week in range(0, 5, 1):
        path = 'input_data/'+paths[week]+'.txt'
        print('Path: ', path)
        price_list = get_price_list(path)
        for i_time in range(1000, 85200, 1):
            state = create_input(i_time, price_list)
            sum_d_perc = 0
            for targ in range(0, target, 1):
                dif_perc = (price_list[i_time+targ][1]-price_list[i_time][1])/price_list[i_time][1]*100
                sum_d_perc += dif_perc
            targ_perc = sum_d_perc/target
            sum_targ_perc += abs(targ_perc)
            z_targ_perc += 1
                
            if targ_perc > hold_perc_threshold:  # Long
                list_train_labels.append([1, 0])
                list_train_states.extend(state)
            elif targ_perc < -hold_perc_threshold:  # Short
                list_train_labels.append([0, 1])
                list_train_states.extend(state)
    print('len(list_train_states)  //  len(list_train_labels): ', len(list_train_states), '  //  ', len(list_train_labels))

    list_test_states, list_test_labels = [], []
    for week in range(5, 6, 1):
        path = 'input_data/'+paths[week]+'.txt'
        print('Path: ', path)
        price_list  =get_price_list(path)
        for i_time in range(1000, 85200, 1):
            state = create_input(i_time, price_list)
            sum_d_perc = 0
            for targ in range(0,target,1):
                dif_perc = (price_list[i_time+targ][1]-price_list[i_time][1])/price_list[i_time][1]*100
                sum_d_perc += dif_perc
            targ_perc = sum_d_perc/target
            sum_targ_perc += abs(targ_perc)
            z_targ_perc += 1
            
            if targ_perc > hold_perc_threshold:  #Long
                list_test_labels.append([1, 0])
                list_test_states.extend(state)
            elif targ_perc < -hold_perc_threshold:  #Short
                list_test_labels.append([0, 1])
                list_test_states.extend(state)
    print('len(list_test_states)  //  len(list_test_labels): ', len(list_test_states), '  //  ', len(list_test_labels))
    print('Average target percent: ', round(sum_targ_perc/z_targ_perc, 4))
    return list_train_states, list_test_states


def scale_to_one(train_st, test_st):
    max_value = .003
    L_Train = []
    for train_2 in train_st:
        Train = []
        for tr_2 in train_2[0]:
            if abs(tr_2) > max_value:
                Train += [1]
            else:
                Train.append(round(tr_2/max_value, 8))
        L_Train.append([Train])
    L_Test = []
    for test_2 in test_st:
        Test = []
        for te_2 in test_2[0]:
            if abs(te_2)>max_value:
                Test += [1]
            else:
                Test.append(round(te_2/max_value, 8))
        L_Test.append([Test])
    return L_Train, L_Test



train_states, test_states = calculate_train_data()
train_states, test_states = scale_to_one(train_states, test_states)
train_states, test_states = normalize_main(train_states), normalize_main(test_states)

input_img = keras.Input(shape=(399,))
encoded = layers.Dense(350, activation='tanh')(input_img)
encoded = layers.Dense(300, activation='tanh')(encoded)
encoded = layers.Dense(250, activation='tanh')(encoded)
encoded = layers.Dense(200, activation='tanh')(encoded)
encoded = layers.Dense(150, activation='tanh')(encoded)
encoded = layers.Dense(100, activation='tanh')(encoded)
encoded = layers.Dense(50, activation='tanh')(encoded)

decoded = layers.Dense(100, activation='tanh')(encoded)
decoded = layers.Dense(150, activation='tanh')(decoded)
decoded = layers.Dense(200, activation='tanh')(decoded)
decoded = layers.Dense(250, activation='tanh')(decoded)
decoded = layers.Dense(300, activation='tanh')(decoded)
decoded = layers.Dense(350, activation='tanh')(decoded)
decoded = layers.Dense(399, activation='sigmoid')(decoded)


autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()
encoder = keras.Model(input_img, encoded)
encoder.summary()

print('BEGINNNING 1: ', train_states[100][50:100])
print()
print('BEGINNNING 2: ', train_states[300][50:100])
print()
print('BEGINNNING 3: ', train_states[500][50:100])
print()

history = autoencoder.fit(train_states, train_states, epochs=300, batch_size=1, shuffle=True, verbose=2,
                          validation_data=(test_states, test_states))
save_model(autoencoder, 1)
save_model(encoder, 2)
print('ENDING 1: ', encoder.predict([train_states[100]]))
print()
print('ENDING 2: ', encoder.predict([train_states[300]]))
print()
print('ENDING 3: ', encoder.predict([train_states[500]]))
print()

print('ENDING AUTO 1: ', autoencoder.predict([train_states[100]])[0][50:100])
print()
print('ENDING AUTO 2: ', autoencoder.predict([train_states[300]])[0][50:100])
print()
print('ENDING AUTO 3: ', autoencoder.predict([train_states[500]])[0][50:100])
plot_train_history(history)
