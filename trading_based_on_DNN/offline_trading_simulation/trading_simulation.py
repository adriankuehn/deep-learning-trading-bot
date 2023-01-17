from keras.models import model_from_json
import matplotlib.pyplot as plt


def calculate_input_data():
    global number_weeks

    def create_input_state(i_timep, price_list):
        faktors = [1, 1, 1, 1, 1, 1, 1]  # Here you can define pattern where input for network should be calculated on: e.g.: [1,2,4,8,16,32,128]
        number = 10
        input_prices, sum_forward = [], 0
        for fak in faktors:
            for i in range(0, number, 1):
                try:
                    input_prices.append(price_list[i_timep - sum_forward - i * fak][1])
                    if i_timep - sum_forward - i * fak < 0:
                        print('Error')
                except:
                    print('Error at timepoint: ', i_timep - sum_forward - i * fak)
            sum_forward += number * fak

        perc_input = []
        for i_p in range(0, len(input_prices) - 1, 1):
            before = input_prices[i_p + 1]
            after = input_prices[i_p]
            perc = (after - before) / before * 100
            perc_input += [round(perc, 6)]
        return perc_input

    def get_price_list_and_marketprice(path):
        data, data_prices = open(path, 'r'), []
        for line in data:
            v1, v2, v3 = line.rstrip().split(' | ')
            time, marketprice = v3.split(' /')
            data_prices.append([int(float(v1)), float(v2), v3, float(marketprice)])
        return data_prices

    list_path = ['28.20','29.20','30.20','31.20','32.20','33.20','34.20','35.20','36.20','37.20']
    list_input_states, list_input_prices_current, number_weeks = [], [], 0
    for week in range(8, 10, 1):  # test data
        path = 'input_data/'+list_path[week]+'.txt'
        print('path: ', path)
        price_list = get_price_list_and_marketprice(path)
        for i_time in range(70, len(price_list), 1):    # 70 = len(faktors) * number
            state = create_input_state(i_time, price_list)
            list_input_states.append(state)
            list_input_prices_current.append(price_list[i_time][3])
        list_input_states.append(['weekend'])
        list_input_prices_current.append(['weekend'])
        number_weeks += 1

    return list_input_states, list_input_prices_current


def calculate_profit_loss(history_trades, current_price):
    global start_capital, sum_lever, z_trades, acc_reward_perc, count_max_lever
        
    value_invested_capital, number_units = 0, 0
    for i_k in range(0, len(history_trades), 1):
        value_invested_capital += history_trades[i_k][1] * history_trades[i_k][2]
        number_units = number_units + history_trades[i_k][2]
        if value_invested_capital > start_capital*max_leverage:  # Theoretical amount of invested money exceeds maximum allowed leverage
            count_max_lever += 1
            break
    value_profit_loss = number_units * current_price - value_invested_capital
    leverage = round(value_invested_capital / start_capital, 3)
    sum_lever += leverage
    z_trades += 1
    
    if history_trades[0][0] == 'Short':
        value_profit_loss = value_profit_loss - spread * number_units
        dif_perc = value_profit_loss / start_capital * 100
        acc_reward_perc = acc_reward_perc + dif_perc
        start_capital = round(start_capital + value_profit_loss, 2)

    elif history_trades[0][0] == 'Long':
        value_profit_loss *= -1
        value_profit_loss = value_profit_loss - spread * number_units
        dif_perc = value_profit_loss / start_capital * 100
        acc_reward_perc = acc_reward_perc + dif_perc
        start_capital = round(start_capital + value_profit_loss, 1)
    print('Cap_Value | Profit | S/L | Leverage: ', start_capital, '  |  ', round(dif_perc, 3), '  |  ', history_trades[0][0], '  |  ', leverage)


def load_model(path):
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path + '.h5')
    return loaded_model
    

def main_trading_simulation(input_states, input_prices_current, model):
    global start_capital, sum_lever, z_trades, acc_reward_perc, count_max_lever, number_weeks

    """ Simulation decides for each timepoint whther to go Long, Short or Hold. If the trading "direction" is changed
     (from long to short or from short to long) all the invested capital is settled and profits are calculated based
      on the current market price """

    list_capital_value, list_capital_indices, list_trades = [], [], []
    previous_trade_direction=999
    z_long, z_short, z_hold = 0, 0, 0
    sum_lever, z_trades = 0, 0
    acc_reward_perc, count_max_lever = 0, 0
    logfile, len_input_states = open('Logfile T_Area.txt', 'w'), len(input_states)
    print('Number of simulated timepoints: ', len_input_states)

    for inp in range(0, len(input_states), 1):
        if inp % 100 == 0: print('Simulation progress (in %): ', round(inp / len_input_states * 100, 1))
        if start_capital < 0:
            print('Stopp, account balance below zero')
            break
        
        state = [[input_states[inp]]]
        value_weekend = 1
        if state != [[['weekend']]]:
            current_market_price = input_prices_current[inp]
            list_capital_value += [start_capital]
            list_capital_indices += [inp]

            prediction = model.predict(state)
            logfile.write('('+str(round(prediction[0][0],3))+', '+str(round(prediction[0][1],3))+')'+'\n')
            if prediction[0][0] > trading_threshold:
                a = 0  # omega takes confidence of network into account and applies it on trading quantity
                omega = round((prediction[0][a] - trading_threshold) / (1 - trading_threshold), 3)
            elif prediction[0][1] > trading_threshold:
                a = 1
                omega = round((prediction[0][a] - trading_threshold) / (1 - trading_threshold), 3)
            else:
                a = 2
                omega = 999
        else: value_weekend = 0

        if (previous_trade_direction==0 and a==1) or (previous_trade_direction==1 and a==0) or value_weekend==0:
            calculate_profit_loss(list_trades, current_market_price)
            list_trades = []
        if a == 0 or value_weekend == 0:
            list_trades.append(['Long', current_market_price, standard_trading_volume * start_capital * omega * value_weekend, state])
            previous_trade_direction = a
        if a == 1:
            list_trades.append(['Short', current_market_price, standard_trading_volume * start_capital * omega * value_weekend, state])
            previous_trade_direction = a
            
        if a == 0: z_long += 1
        elif a == 1: z_short += 1
        elif a == 2: z_hold += 1

    print()
    print('Proportions =>   Long-Trades:', round(z_long / (z_long + z_short + z_hold), 3), '   | Short Trades:', round(z_short / (z_long + z_short + z_hold), 3), '   | Hold Trades:', round(z_hold / (z_long + z_short + z_hold), 3))
    print('Average leverage used  ||  Number trades: ', round(sum_lever / z_trades, 3), '  ||  ', z_trades)
    print('Average reward per trade in percent: ', round(acc_reward_perc / z_trades, 4))
    print('Number of times the maximum leverage was reached: ', count_max_lever)
    print()
    print()
    logfile.write('\n')
    logfile.close()
    plt.plot(list_capital_indices, list_capital_value, 'b', color='mediumblue')
    plt.plot([0, list_capital_indices[len(list_capital_indices)-1]], [list_capital_value[0],list_capital_value[0]], 'b', color='lightgray')
    week_size = len_input_states / number_weeks * 0.995
    for xx in range(0, int(list_capital_indices[len(list_capital_indices)-1]/week_size)+1, 1):
        plt.plot([week_size*xx, week_size*xx], [list_capital_value[0]*0.93,list_capital_value[0]*1.07], 'b', color='gray')
    plt.show()



trading_threshold = 0.505
max_leverage = 25  # Defines the maximum allowed leverage which the trading simulation can utilize
start_capital = 50000
standard_trading_volume = 5
spread = 0.000005   # in pips = 0.0001, dif between bid and ask

model = load_model('model_final_15')
input_states, input_prices_current = calculate_input_data()
main_trading_simulation(input_states, input_prices_current, model)
