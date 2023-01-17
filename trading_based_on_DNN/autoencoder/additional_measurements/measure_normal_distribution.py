import matplotlib.pyplot as plt


def get_price_list(path):
    data, data_prices = open(path, 'r'), []
    for line in data:
        v1, v2, v3 = line.rstrip().split(' | ')
        data_prices.append([int(float(v1)), float(v2), v3])
    return data_prices
                  

def calculation():
    path = 'input_data/37.20.txt'
    price_list = get_price_list(path)
    sum_var, sum_dif, z_null = 0, 0, 0
    for i_pr in range(0, len(price_list)-1, 1):
        var = abs(price_list[i_pr][1] - price_list[i_pr+1][1]) / price_list[i_pr][1]
        dif = abs(price_list[i_pr][1] - price_list[i_pr+1][1])
        sum_var += var
        sum_dif += dif
        if var <= list_distribution[0]:
            z_null += 1
        for ii in range(0, len(list_distribution) - 1, 1):
            if var>list_distribution[ii] and var<=list_distribution[ii + 1]:
                list_normdis_values[ii] += 1

    print('Average deviation in %: ', round(sum_var/(len(price_list)-1),6))
    print('Average difference in %: ', round(sum_dif/(len(price_list)-1),6))
    print('z_null: ', z_null)

list_distribution = []
devisor = 1000000
range_d = 100
for x in range(1, range_d, 1): list_distribution.append(x / devisor)

list_normdis_values= [0] * len(list_distribution)
calculation()
print('list_normdis_values: ', list_normdis_values)
plt.plot(list_distribution, list_normdis_values, color='blue')
plt.show()
