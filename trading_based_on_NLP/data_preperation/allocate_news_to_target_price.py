import numpy as np

# list_allocation structure: [[[date, time], headline, trailtext, body, [t1, t2, t3, ... t20]], ...]
# back = time in seconds to go back before the message appears
# forward = time in seconds which is advanced after the message appears
# average = size from the period to the forward time from which the average target value is calculated
# target_params structure:  Back=0, Forward=2, Average=1

target_params = [[0, 2, 1], [0, 5, 1], [0, 20, 3], [0, 60, 10], [0, 900, 90],
                 [5, 2, 1], [5, 5, 1], [5, 20, 3], [5, 60, 10], [5, 900, 90],
                 [30, 2, 1], [30, 5, 1], [30, 20, 3], [30, 60, 10], [30, 900, 90],
                 [300, 2, 1], [300, 5, 1], [300, 20, 3], [300, 60, 10], [300, 900, 90]]


def get_targets(date, time):
    # always only within one day, not daily calculations possible
    global target_params, eur_usd_array, count_incomplete

    def convert_back(ind):
        hour = str(int(ind / 3600))
        if len(hour) == 1: hour = '0'+hour
        min_c = str(int((ind - int(ind / 3600) * 3600) / 60))
        if len(min_c) == 1: min_c = '0'+min_c
        sek = str(int(ind - int(ind / 3600) * 3600 - int((ind - int(ind / 3600) * 3600) / 60) * 60))
        if len(sek)==1: sek = '0'+sek
        return hour+':'+min_c+':'+sek
    
    list_targets_t = []
    v_incomplete = False
    for tar in target_params:
        index_in= int(time[0:2]) * 3600 + int(time[3:5]) * 60 + int(time[6:8]) - tar[0]
        time_in = convert_back(index_in)   # mon, day, hour, min, sek
        position_in = (int(date[0:2]) - 1, int(date[3:5]) - 1, int(time_in[0:2]) - 1, int(time_in[3:5]) - 1, int(time_in[6:8]) - 1)
        entry_price = eur_usd_array[position_in]  # price at which trading bot can start position
        
        sum_exitp = 0
        out_of_day = False
        no_price_available = False
        for i in range(0, tar[2], 1):
            index_out = int(time[0:2]) * 3600 + int(time[3:5]) * 60 + int(time[6:8]) - tar[0] + tar[1] + i
            if index_out < 0 or index_out > 86400:
                out_of_day = True
            else:
                time_out = convert_back(index_out)
                position_out = (int(date[0:2]) - 1, int(date[3:5]) - 1, int(time_out[0:2]) - 1, int(time_out[3:5]) - 1,
                                int(time_out[6:8]) - 1)
                if eur_usd_array[position_out] != 0:
                    sum_exitp += eur_usd_array[position_out]
                else:
                    no_price_available=True
                
        if out_of_day == False and entry_price != 0 and no_price_available == False:
            exit_price = sum_exitp / tar[2]       # Price at which Trading Bot could theoretically close position
            perc_dif = (exit_price - entry_price) / entry_price * 100
            list_targets_t.append(round(perc_dif, 5))
        else:
            list_targets_t.append('99')
            v_incomplete = True
    if v_incomplete: count_incomplete += 1
    return list_targets_t


def get_dt(line_g):
    date = line_g[0:10]                                 # Jahr-Monat-Tag  2020-12-29
    end_time = line_g[11:19]                            # h:min:s   18:23:30
    end_date = date[5:7]+'/'+date[8:10]+'/'+date[0:4]   # Monat/Tag/Jahr  01/19/2020
    return end_date, end_time


def perform_allocation(section):
    global count_incomplete

    def check_all_99(list_targets):
        for tar in list_targets:
            if tar !='99':
                return False
        return True
    
    list_allocation, count_incomplete = [], 0
    news_array = np.load("news_2020_as_array/business_2020_object.npy", allow_pickle=True)
    print('news_array.shape: ', news_array.shape)
    for news in news_array:
        date, time = get_dt(news[0])
        list_targets = get_targets(date, time)
        if not check_all_99(list_targets):
            list_allocation.append([[date, time], news[1], news[2], news[3], list_targets])

    print('count_incomplete: ', count_incomplete)
    print('number complete news which can be used for training (where target price could be allocated): ', news_array.shape[0] - count_incomplete)
    array_allocation = np.array(list_allocation, dtype=object)
    np.save("result_prep_data/allocated_"+section, array_allocation, allow_pickle=True)
    print('saved')
    data_loaded = np.load("result_prep_data/Allocated_"+section+".npy", allow_pickle=True)
    print('loaded - everything works fine')

eur_usd_array = np.load("eur_usd_array_object.npy", allow_pickle=True)
perform_allocation("business")
