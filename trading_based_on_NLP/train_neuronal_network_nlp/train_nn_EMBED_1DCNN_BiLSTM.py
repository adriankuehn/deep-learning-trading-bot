import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer


def recover_text(word_i, sequences_r):
    rev_word_i = dict([(v, k) for (k, v) in word_i.items()])
    list_text = []
    for seq in sequences_r:
        text = ""
        for ss in seq: text += rev_word_i[int(ss)+1] + " "
        list_text.append(text[0:-1])
    return list_text


def plot_train_history(history_p):
    acc = history_p.history['acc']
    val_acc = history_p.history['val_acc']
    loss = history_p.history['loss']
    val_loss = history_p.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def visualize_label_distribution(labels):
    labels = np.array(labels)
    print('mean: ', np.mean(labels))
    print('std: ', np.std(labels))
    print('max, min: ', max(labels), min(labels))
    plt.hist(labels, bins=100, density=True)
    plt.show()


def get_traindata_labels(data, tar_number):
    """ Data = Headline & Trailtext from news
        Label = Pricetarget, whether price goes up or down after web publication date of news """
    train_d, labels = [], []
    for dd in data:
        if dd[4][tar_number]!= '99':
            Text = dd[1]+' '+dd[2]
            train_d.append(Text)
            labels.append(dd[4][tar_number])
            
    visualize_label_distribution(labels)
    threshold = 0.0012
    labels_new = []
    train_d_new = []
    z_sell = 0
    z_buy = 0
    for ii in range(0, len(labels), 1):
        if labels[ii] < (-threshold):  # sell -> go short
            labels_new.append(0)
            train_d_new.append(train_d[ii])
            z_sell += 1
        elif labels[ii] > threshold:  # buy -> go long
            labels_new.append(1)
            train_d_new.append(train_d[ii])
            z_buy += 1
    print('length data: ', len(labels_new), 'z_sell: ', z_sell, 'z_buy: ', z_buy)
    return np.array(train_d_new), np.array(labels_new)


def fill_sequence_with_nulls_based_on_aver_length(sequences_f, aver_len_f):

    def get_aver_len(g_sequences):
        g_sum=0
        for g_seq in g_sequences:
            g_sum += len(g_seq)
        return round(g_sum / len(g_sequences), 2)

    g_aver_len = get_aver_len(sequences_f)   # 20 words were used in average news text, Aver=17.8
    print('Average Length in number of words: ', g_aver_len)
    
    seq_new = []
    for seq in sequences_f:
        seq_n = []
        for ss in seq:
            seq_n.append(ss)
            if len(seq_n) == aver_len_f: break
        for ii in range(0, aver_len_f - len(seq_n), 1): seq_n.append(0)
        seq_new.append(seq_n)
    return seq_new


def save_model(model):
    index_model = '__End__'
    model_json = model.to_json()
    with open("model" + str(index_model) + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model" + str(index_model) + ".h5")


        
aver_len = 20
number_words_used = 1000

data_array = np.load('Allocated_business_2020.npy', allow_pickle=True)
maindata, labels = get_traindata_labels(data_array, 5)
tokenizer = Tokenizer(num_words=number_words_used)                       # 1000 most common words were used
tokenizer.fit_on_texts(maindata)                                         # build the word index
one_hot_DATA = tokenizer.texts_to_matrix(maindata, mode='binary')        # convert into one-hot binary representations
sequences = tokenizer.texts_to_sequences(maindata)                       # convert into list with integer indices
sequences = fill_sequence_with_nulls_based_on_aver_length(sequences, aver_len)
word_index = tokenizer.word_index
print('number of unique tokens: ', len(word_index))
# print('word_index: ', word_index)

train_data = np.array(sequences[0: int(len(sequences) * 0.75)])
test_data = np.array(sequences[int(len(sequences) * 0.75): len(sequences)])
train_labels = np.array(labels[0: int(len(sequences) * 0.75)])
test_labels = np.array(labels[int(len(sequences) * 0.75): len(sequences)])

print('recovered text: ', recover_text(word_index, sequences))
print('train_data[0:10]: ', train_data[0:10])
print('train_labels[0:10]: ', train_labels[0:10])

model = models.Sequential()
model.add(layers.Embedding(number_words_used, 32, input_length=aver_len))
model.add(layers.Conv1D(128, 5, activation='sigmoid'))
# model.add(layers.MaxPooling1D(3))
# model.add(layers.Conv1D(64, 5, activation='sigmoid'))
model.add(layers.Bidirectional(layers.LSTM(18, return_sequences=True, recurrent_dropout=0.1)))
# model.add(layers.LSTM(32, recurrent_dropout=0.1))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels), verbose=2, shuffle=True)
save_model(model)
plot_train_history(history)
