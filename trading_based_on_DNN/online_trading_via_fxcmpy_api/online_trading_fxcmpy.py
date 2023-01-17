import time
import os
from keras.models import model_from_json
import matplotlib.pyplot as plt
from _thread import start_new_thread
import fxcmpy


def create_input_state(price_list):

    def reverse(list_rev):
        list_new = []
        for ll in range(len(list_rev) - 1, -1, -1): list_new.append(list_rev[ll])
        return list_new
    
    def calculate_average_per_step(list_time_price, step):
        list_tp_aver = []
        remainder = len(list_time_price) - int(len(list_time_price) / step) * step
        for lz in range(len(list_time_price) - 1, remainder - 1, -step):
            aver = 0
            for ss in range(0, step, 1):
                if lz-ss < 0:
                    print('Error lz-ss negativ:', lz-ss)
                    lop
                aver += list_time_price[lz - ss][1]
            list_tp_aver.append([list_time_price[lz][0], round(aver / step, 6), list_time_price[lz][2]])

        list_tp_aver = reverse(list_tp_aver)
        return list_tp_aver

    def create_input(timep_index, list_time_price):
        factors = [1,1,1,1]  # Here you can define pattern where input for network should be calculated on: e.g.: [1,2,4,8,16,32,128]
        number, c_input, c_sum = 10, [], 0
        for fak in factors:
            for i in range(0, number, 1):
                try:
                    c_input.append(list_time_price[timep_index - c_sum - i * fak][1])
                    if timep_index-c_sum-i*fak<0: lop
                except:
                    print('Position: ', timep_index - c_sum - i * fak)
                    lop
            c_sum += number*fak
        perc_input = []
        for i_p in range(0, len(c_input)-1, 1):
            before = c_input[i_p+1]
            after = c_input[i_p]
            perc = (after-before)/before*100
            perc_input += [round(perc,6)]
        return [[perc_input]]

    def fill_each_second_with_one_price(list_time_price):
        new_list_tp=[list_time_price[0]]
        for i_tp in range(0, len(list_time_price) - 1, 1):
            if list_time_price[i_tp + 1][0]!=list_time_price[i_tp][0]:
                new_list_tp.append(list_time_price[i_tp + 1])
        
        filled_list_tp = []
        for i_neu in range(0, len(new_list_tp)-1, 1):
            filled_list_tp.append(new_list_tp[i_neu])
            if new_list_tp[i_neu+1][0] != new_list_tp[i_neu][0]:
                for iii_n in range(1, int(new_list_tp[i_neu+1][0]-new_list_tp[i_neu][0]), 1):
                    v = new_list_tp[i_neu][0]+iii_n
                    v_dec = str(int(v/3600))+':'+str(int((v-int(v/3600)*3600)/60))+':'+str(int(v-int(v/3600)*3600-int((v-int(v/3600)*3600)/60)*60))
                    filled_list_tp.append([v, new_list_tp[i_neu][1], v_dec])
        filled_list_tp.append(new_list_tp[len(new_list_tp)-1])
        return filled_list_tp
    
    price_list.sort()
    price_list = fill_each_second_with_one_price(price_list)
    price_list = calculate_average_per_step(price_list, 5)
    state_final = create_input(len(price_list) - 1, price_list)
    return state_final


def load_model(path):
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path + '.h5')
    return loaded_model


def update_price_list(con):
    global glob_price_list, glob_last_update, glob_z_long, glob_z_short, glob_z_hold, glob_list_capital_indeces, glob_list_capital_value, glob_additional_tag

    def get_time(pd_time):
        z_pd, start_name = 0, False
        for pd in pd_time:
            part_1 = pd_time[z_pd - 3:z_pd + 1]
            if part_1 =="2023": start_name=True
            if start_name:
                if pd==':':
                    time_final= pd_time[z_pd - 2:z_pd + 6]
                    return time_final
            z_pd += 1
        return False

    def time_to_deci(d_time): return float(d_time[0:2]) * 60 * 60 + float(d_time[3:5]) * 60 + float(d_time[6:8])

    def last_15_prices_equal(price_list):
        last_timepoint = price_list[len(price_list) - 1][0]
        last_price = price_list[len(price_list) - 1][1]
        for zpp in range(len(price_list) - 1, len(price_list) - 16, -1):
            if last_timepoint != price_list[zpp][0]: return False
            if last_price != price_list[zpp][1]: return False
        return True
    
    con.subscribe_market_data(currency)
    Z_error = 0
    while Z_error < 20:
        if controll_action(con) == 'Break':
            break
        if len(glob_price_list) > 15 and last_15_prices_equal(glob_price_list):
            con.unsubscribe_market_data(currency)
            con.subscribe_market_data(currency)
            print()
            print('Subscribtion renewed!')
            print()

        last_price = con.get_last_price(currency)
        time_cur = get_time(str(last_price))
        if len(glob_price_list) != 0 and abs(time_to_deci(time_cur) - glob_price_list[len(glob_price_list) - 1][0]) > 5000:
            glob_additional_tag += 86400
            print()
            print('***** Additional tag increased to: ', glob_additional_tag, '*****')
            print('New tag')
            print()
        time_dec = time_to_deci(time_cur) + glob_additional_tag  # in s
        bid = last_price["Bid"]
        ask = last_price["Ask"]
        price = round((bid+ask)/2, 6)

        if len(glob_price_list) % 300 == 0 and len(glob_price_list) != 0:
            print('Length price_list | time | price:    ', len(glob_price_list), ' | ', time_cur, ' | ', price)
        glob_price_list.append([time_dec, price, time_cur])
        data_stream.write(str(time_dec) + ' | ' + str(price) + ' | ' + str(time_cur) + '\n')
        Z_error = 0
        time.sleep(0.9)  # save computational resources
        glob_last_update = time.time()
        
        if time.time()-glob_last_update > 20:
            print('Could not receive any price update from server in the last 20 seconds')
            con.close_all_for_symbol(currency)
            con.close()
            evaluation(glob_z_long, glob_z_short, glob_z_hold, glob_list_capital_indeces, glob_list_capital_value)

    print('Exceeded maximum number of errors which is 20!')
    con.close_all_for_symbol(currency)
    con.close()
    evaluation(glob_z_long, glob_z_short, glob_z_hold, glob_list_capital_indeces, glob_list_capital_value)
            
    
def evaluation(z_long, z_short, z_hold, list_capital_index, list_capital_value):
    logfile.write('\n')
    logfile.close()
    data_stream.close()
    plt.plot(list_capital_index, list_capital_value, 'b', color='mediumblue')
    plt.plot([list_capital_index[0], list_capital_index[len(list_capital_index) - 1]], [list_capital_value[0],
                                                                                        list_capital_value[0]], 'b', color='lightgray')
    plt.show()
    print()
    print('Proportions =>  Long:', round(z_long / (z_long + z_short + z_hold), 3), '  | Short:',
          round(z_short / (z_long + z_short + z_hold), 3), '  | Hold:', round(z_hold / (z_long + z_short + z_hold), 3))
    print()


def controll_action(con):
    data = os.listdir()
    for dd in data:
        if dd == 'Stop.txt':
            con.close_all_for_symbol(currency)
            print('Trading was stopped')
            return 'Stop'
            
        if dd == 'Break.txt':
            con.close_all_for_symbol(currency)
            con.close()
            print('Programm was stopped')
            return 'Break'


def online_trading():
    global glob_price_list, glob_last_update, glob_z_long, glob_z_short, glob_z_hold, glob_list_capital_indeces,\
        glob_list_capital_value, glob_additional_tag

    hold_threshold = 0.66
    standard_tradevolume = 80  # in k also 4 Lots per trade
    glob_price_list, glob_list_capital_value, glob_list_capital_indeces = [], [], []
    previous_trade_direction = 999
    glob_z_long, glob_z_short, glob_z_hold = 0, 0, 0
    sum_buy, sum_sell, error_counter = 0, 0, 0
    glob_additional_tag, glob_last_update = 0, time.time()
    start_new_thread(update_price_list, (con,))

    while True:
        if len(glob_price_list)!=0 and glob_price_list[0][0]+220 < glob_price_list[len(glob_price_list) - 1][0]:
            break
        time.sleep(1)
    old_length = len(glob_price_list) - 1

    print()
    print('Start Trading!')
    print()
    while True:
        trading = True
        if controll_action(con) == 'Break':                                                                                         #Programm wird abgebrochen
            break
        elif controll_action(con) == 'Stop':                                                                                        #Handel wird ausgesetzt
            trading = False
            
        
        if glob_price_list[old_length][0]+ 5 <= glob_price_list[len(glob_price_list) - 1][0] and trading:
            try:
                old_length = len(glob_price_list) - 1
                state = create_input_state(glob_price_list)
                equity_value = con.get_accounts(kind='list')[0].get('equity')
                if equity_value < 6000:
                    con.close_all_for_symbol(currency)
                    print('Close to all capital lost, positions were closed and trading was stopped')
                
                glob_list_capital_value.append(equity_value)
                glob_list_capital_indeces.append(glob_price_list[len(glob_price_list) - 1][0])
                
                prediction = model.predict(state)
                print('Current time | Price | Prediction:  ', glob_price_list[len(glob_price_list) - 1][2], ' | ',
                      glob_price_list[len(glob_price_list) - 1][1], ' | ', [round(prediction[0][0], 3), round(prediction[0][1], 3)])
                logfile.write('(' + str(round(prediction[0][0], 3)) + ', ' + str(round(prediction[0][1], 3)) + ')' + '\n')
                if prediction[0][0] > hold_threshold:
                    a = 0
                    omega = round((prediction[0][a]-hold_threshold)/(1-hold_threshold),3)
                elif prediction[0][1] > hold_threshold:
                    a = 1
                    omega = round((prediction[0][a]-hold_threshold)/(1-hold_threshold),3)
                else:
                    a = 2
                    omega = '999'

                if (previous_trade_direction == 0 and a == 1) or (previous_trade_direction == 1 and a == 0):
                    con.close_all_for_symbol(currency)
                    print('Equity: ', equity_value, '  |  Margin at 50%: ', round(con.get_accounts(kind='list')[0].get('usableMargin')/equity_value*100,2), \
                          '  | Leverage: ', round((sum_buy+sum_sell) * glob_price_list[len(glob_price_list) - 1][1] / equity_value * 1000, 2), '#####')
                    sum_buy, sum_sell = 0, 0
                    
                security_margin = equity_value/16000*430  # 10% security threshold
                if a == 0:
                    if standard_tradevolume*omega+sum_buy < security_margin:
                        order = con.create_market_buy_order(currency, standard_tradevolume * omega)
                        print('buy: ', round(standard_tradevolume*omega, 2))
                        sum_buy += standard_tradevolume*omega
                        previous_trade_direction = a
                if a == 1:
                    if standard_tradevolume*omega+sum_sell < security_margin:
                        order = con.create_market_sell_order(currency, standard_tradevolume * omega)
                        print('sell: ', round(standard_tradevolume*omega, 2))
                        sum_sell += standard_tradevolume*omega
                        previous_trade_direction = a
                    
                if a == 0: glob_z_long += 1
                elif a==1: glob_z_short += 1
                elif a==2: glob_z_hold += 1
                    
            except Exception as e:
                print('Try-Except error occured: ', e)
                error_counter += 1
                if error_counter >= 2:
                    print('Two errors occured in trading area, trading stopped, tried to close all positions')
                    con.close_all_for_symbol(currency)
                    lop

    evaluation(glob_z_long, glob_z_short, glob_z_hold, glob_list_capital_indeces, glob_list_capital_value)



model = load_model('model__Ende__')
data_stream = open('Data_Stream.txt', 'w')
logfile = open('Logfile_T_Area.txt', 'w')
token = '***************************************'
con = fxcmpy.fxcmpy(access_token=token, log_level='error', server='demo', log_file='log.txt')
print('Delta One: Model loaded and Broker-Connection aktive || Ready to trade')

currency= 'EUR/USD'
# survives 10s without active internet connection, afterwards trading brakes up
online_trading()

con.close_all_for_symbol(currency)
con.close()
logfile.close()
data_stream.close()


