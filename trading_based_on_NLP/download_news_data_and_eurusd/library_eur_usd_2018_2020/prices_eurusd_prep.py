import os


def create_list_tp(path):

    def convert_time_to_decim(time): return float(time[0:2]) * 60 * 60 + float(time[3:5]) * 60 + float(time[6:8])

    data = open(path, 'r')
    list_time_price, z_line = [], 0
    for line in data:
        if z_line > 0:
            date, time_prices = line.rstrip().split(' ')
            time_miliseconds, bid, ask = time_prices.split(',')
            time_seconds = time_miliseconds.split('.')[0]
            list_time_price.append([float(convert_time_to_decim(time_seconds)), float(bid), time_seconds, date])
        z_line += 1
    return list_time_price


def filter_list_tp(list_tp):
    new_list_tp=[]
    for i_tp in range(0, len(list_tp) - 1, 1):
        if list_tp[i_tp + 1][0] != list_tp[i_tp][0]:
            new_list_tp.append(list_tp[i_tp])
            
    filled_list_tp = []
    for i_neu in range(0, len(new_list_tp)-1, 1):
        filled_list_tp.append(new_list_tp[i_neu])
        
        if new_list_tp[i_neu+1][0] != new_list_tp[i_neu][0]:
            if new_list_tp[i_neu+1][0] > new_list_tp[i_neu][0]:
                run = int(new_list_tp[i_neu+1][0]-new_list_tp[i_neu][0])
            else:
                run = int(new_list_tp[i_neu+1][0]+86400-new_list_tp[i_neu][0])
            if run <= 30:
                for i_n in range(1, run, 1):
                    v = new_list_tp[i_neu][0]+i_n
                    v_date = new_list_tp[i_neu][3]
                    if v>86400:
                        v -= 86400
                        print('v_date old: ', v_date)
                        if len(str(int(v_date[3:5])))==2:
                            v_date = v_date[0:3]+str(int(v_date[3:5])+1)+v_date[5:10]               #Monatssprung und Jahressprung wird nicht berücksichtigt
                        elif len(str(int(v_date[3:5])))==1:
                            v_date = v_date[0:3]+"0"+str(int(v_date[3:5])+1)+v_date[5:10]           #Damit 0 Formatierung auch bei Tag stimmt, nicht nur bei Uhrzeit
                        print('V_Datum NEU: ', v_date)

                    hour = str(int(v/3600))
                    if len(hour) == 1: hour = '0'+hour
                    min_v = str(int((v-int(v/3600)*3600)/60))
                    if len(min_v) == 1: min_v = '0'+min_v
                    sek = str(int(v-int(v/3600)*3600-int((v-int(v/3600)*3600)/60)*60))
                    if len(sek) == 1: sek = '0'+sek
                    v_dez = hour+':'+min_v+':'+sek
                    filled_list_tp.append([v, new_list_tp[i_neu][1], v_dez, v_date])
    return filled_list_tp


def write_list_tp(list_tp, path):
    data_w = open('result/' + path, 'w')
    for i in range(0, len(list_tp), 1):
        data_w.write(str(list_tp[i][3]) + ' | ' + str(list_tp[i][2]) + ' | ' + str(list_tp[i][1]) + '\n')


folder = 'raw_data'
for data in os.listdir(folder):
    print('data: ', data)
    list_time_price = create_list_tp(folder + '/' + data)
    # filter a price for each second, fill non-available seconds with old prices
    list_time_price = filter_list_tp(list_time_price)
    write_list_tp(list_time_price, data)

print()
print('Preperation finished')
