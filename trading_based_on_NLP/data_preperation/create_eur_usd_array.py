import numpy as np
import os

array = np.zeros([12, 31, 24, 60, 60], dtype=object)

for path in os.listdir("eur_usd_2020"):
    print('path: ', path)
    data = open("eur_usd_2020/" + path, 'r')
    data_linelist = data.readlines()
    print('Number of price points: ', len(data_linelist))

    for line in data_linelist:
        try:
            array[int(line[0:2]) - 1, int(line[3:5]) - 1, int(line[13:15]) - 1, int(line[16:18]) - 1, int(line[19:21]) - 1] = float(line[24:len(line)])
        except:
            print('Error line: ', line)
            lop

np.save('eur_usd_array_object.npy', array)
print('Finished')
