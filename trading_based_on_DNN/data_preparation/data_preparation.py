import os


def create_list_zp(path):

    def convert_time_to_decim(time):
        return float(time[0:2]) * 60 * 60 + float(time[3:5]) * 60 + float(time[6:8])
    
    data = open(path, 'r')
    list_time_price, z_line = [], 0
    for line in data:
        if z_line>0:
            time_miliseconds, bid, ask = line.rstrip().split(' ')[1].split(',')
            time_seconds = time_miliseconds.split('.')[0]
            list_time_price.append([float(convert_time_to_decim(time_seconds)), float(bid), time_seconds])
        z_line+=1
    return list_time_price


def calculate_average_diffrence_between_steps(input_list, step):
    z_start = 0
    for ll in input_list:  # get beginning of first step in timeline
        if ll[0] % step == 0: break
        z_start += 1

    sum_perc_dif = 0
    z_perc, z_diff=0, 0
    max_diff, min_diff = 0, 1
    for i_ll in range(z_start, len(input_list) - step, step):
        perc = abs(input_list[i_ll+step][1] - input_list[i_ll][1]) / input_list[i_ll][1] * 100  # calculate percentage diffrence between steps
        if perc > max_diff: max_diff = perc
        elif perc < min_diff: min_diff = perc
            
        sum_perc_dif += perc
        z_perc += 1
        if (input_list[i_ll + step][1] - input_list[i_ll][1])!=0:
            z_diff += 1

    print('Proportion of Changes between steps and total number of steps (in %): ', round(z_diff/z_perc*100, 2))
    print('Average difference between steps (in %): ', round(sum_perc_dif / z_perc, 3))
    print('Max, min difference (in %): ', round(max_diff, 4), ', ', round(min_diff, 4))


def fill_list_and_calculate_average(list_tp, step):
    filtered_list_tp = []
    for i_tp in range(0, len(list_tp) - 1, 1):  # filter one price for each available second
        if list_tp[i_tp + 1][0]!=list_tp[i_tp][0]:
            filtered_list_tp.append(list_tp[i_tp])

    filled_list_tp = []
    for i_new in range(0, len(filtered_list_tp)-1, 1):  # fill all non-available seconds with previous prices
        filled_list_tp.append(filtered_list_tp[i_new])
        if filtered_list_tp[i_new+1][0] != filtered_list_tp[i_new][0]:
            for iii_n in range(1, int(filtered_list_tp[i_new+1][0]-filtered_list_tp[i_new][0]), 1):
                t_fill = filtered_list_tp[i_new][0]+iii_n
                t_fill_dec = str(int(t_fill/3600)) + ':' + str(int((t_fill-int(t_fill/3600)*3600)/60)) + ':' + \
                             str(int(t_fill-int(t_fill/3600)*3600-int((t_fill-int(t_fill/3600)*3600)/60)*60))
                filled_list_tp.append([t_fill, filtered_list_tp[i_new][1], t_fill_dec])

    print('Number filled seconds: ', len(filled_list_tp) - len(filtered_list_tp))
    calculate_average_diffrence_between_steps(filled_list_tp, step)
    filled_list_tp = calculate_average_price_per_step(filled_list_tp, step)
    return filled_list_tp


def save_list(input_list, path):
    Data_w=open('result/' + path, 'w')
    for i in range(0, len(input_list), 1):
        Data_w.write(str(input_list[i][0]) + ' | ' + str(input_list[i][1]) + ' | ' + str(input_list[i][2]) + ' /' +
                     str(input_list[i][3]) + '\n')


def calculate_average_price_per_step(input_list, step):
    list_tp_aver=[]
    for i_tp in range(step, len(input_list) - step - step_future, step):
        aver=0
        for i_s in range(0, step, 1): aver += input_list[i_tp - i_s][1]
        list_tp_aver.append([input_list[i_tp][0], round(aver / step, 6), input_list[i_tp][2], input_list[i_tp + step_future][1]])
    return list_tp_aver


step=150          # Number of seconds used for each average price point (past)
step_future=0     # Number of seconds used to go into future for final market price

Folder='raw_data'
for file in os.listdir(Folder):
    if file[-4:]!='.zip':
        print('File: ', file)
        list_time_price = create_list_zp(Folder + '/' + file)
        number_seconds_available = len(list_time_price)
        list_time_price = fill_list_and_calculate_average(list_time_price, step)       # determine for each second a price and calulate the average price for each step
        save_list(list_time_price, file)
        print()

print('Data preperation finished')
