from keras import models
from keras import layers
import matplotlib.pyplot as plt

    
def plot_train_history(history):
    acc = history.history['loss']
    val_acc = history.history['val_loss']
    Z_epochs = range(1, len(acc) + 1)
    plt.plot(Z_epochs, acc, 'b', label='Training loss', color='red')
    plt.plot(Z_epochs, val_acc, 'b', label='Validation loss', color='blue')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()


def normalize_main(states):
    
    def scale_to_one(stat):
        return stat

    def normalize(stat):
        list_stat = []
        for st in range(0, len(stat), 1):
            norm = round(0.5 + (stat[st] / 2), 8)
            list_stat.append(norm)
        return list_stat
    
    list_norm = []
    for state in states:
        state_n = scale_to_one(state)  # scale to -1 to +1
        state_n = normalize(state_n)
        list_norm.append(state_n)
    
    return list_norm, list_norm, list_norm, list_norm


def denormalize(state):
    list_denom = []
    for st in range(0, len(state), 1):
        list_denom.append(round((state[st]-0.5) *2, 8))
    return list_denom


def calculate_difference(target, prediction):
    sum_dif, sum_tar = 0, 0
    for tt in range(0, len(target), 1):
        sum_dif += abs(target[tt] - prediction[tt])
        sum_tar += abs(target[tt])
    # Average deviation and average size of all values
    return [round(sum_dif / len(target), 5), round(sum_tar / len(target), 5)]
    

def save_model(model):
    model_json = model.to_json()                                                                                          
    with open("model"+".json", "w") as json_file: json_file.write(model_json)
    model.save_weights("model"+".h5")



dataset = [[0, -0.1, 0.8, 0.4, 0.5, -0.5, 0.5, 1], [0.5, -0.5, 0.9, 0.3, 0.2, 0, 1, -0.4], [0, -0.9, 0.3, 0.3, 0.3, 1, -0.1, 0.1],
           [0.1,0.1,-0.2,0.3,1,-0.5,0.7,-0.3], [-0.2,-0.2,0,1,0,0,-0.6,0.2], [0.2,0.2,-1,-0.9,-0.3,-0.4,-0.4,0.4],
           [0.1,1,-0.1,0.1,-0.3,0.2,0.6,-0.2], [1,0.2,-0.8,0.2,-0.1,-0.3,-0.4,0.1]]
train_States, train_labels, test_States, test_labels = normalize_main(dataset)

model = models.Sequential()
model.add(layers.Dense(8, input_dim=8, activation='tanh'))   #
model.add(layers.Dense(7, activation='tanh'))
model.add(layers.Dense(6, activation='tanh'))
model.add(layers.Dense(5, activation='tanh'))
model.add(layers.Dense(4, activation='tanh'))
model.add(layers.Dense(5, activation='tanh'))
model.add(layers.Dense(6, activation='tanh'))
model.add(layers.Dense(7, activation='tanh'))
model.add(layers.Dense(8, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])                                  
model.summary()

history = model.fit(train_States, train_labels, epochs=3000, batch_size=8, shuffle=False, verbose=2, validation_data=(test_States, test_labels))

print()
print('dataset[0]: ', dataset[0])
print('model.predict([train_States[0]])[0]: ', model.predict([train_States[0]])[0])
print('denormalized: ', denormalize(model.predict([train_States[0]])[0]))
print()
pred_0 = denormalize(model.predict([train_States[0]])[0])
pred_2 = denormalize(model.predict([train_States[2]])[0])
pred_4 = denormalize(model.predict([train_States[4]])[0])
pred_6 = denormalize(model.predict([train_States[6]])[0])
print('0: ', pred_0, 'Difference %: ', calculate_difference(dataset[0], pred_0))
print('2: ', pred_2, 'Difference %: ', calculate_difference(dataset[2], pred_2))
print('4: ', pred_4, 'Difference %: ', calculate_difference(dataset[4], pred_4))
print('6: ', pred_6, 'Difference %: ', calculate_difference(dataset[6], pred_6))

sum_dif, sum_tar = 0, 0
for norm in range(0, len(dataset), 1):
    pred_ = denormalize(model.predict([train_States[norm]])[0])
    dif = calculate_difference(dataset[norm], pred_)
    sum_dif += dif[0]
    sum_tar += dif[1]
print()
print('Average deviation: ', round(sum_dif / len(dataset), 5), round(sum_tar / len(dataset), 5))

save_model(model)
plot_train_history(history)
