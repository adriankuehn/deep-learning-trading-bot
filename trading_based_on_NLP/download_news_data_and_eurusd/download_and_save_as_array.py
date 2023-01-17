from bs4 import BeautifulSoup
import requests
import numpy as np


section= "politics"
MY_API_KEY = "************************************"
API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {'from-date': "2020-01-01", 'to-date': "2020-12-31", 'section' : section, 'show-fields': 'headline,trailText,body', 'page-size': 200,
             'page':1, 'type':'article','api-key': MY_API_KEY}

z_page, collector_list = 1, []
while z_page <= 15:
    # 2018-2020 -> 58-53 and for 2020 -> 15-20
    print('page: ', z_page)
    my_params['page'] = z_page
    resp = requests.get(API_ENDPOINT, my_params)
    data = resp.json()
    if z_page == 1:
        print('Total: ', data['response']['total'])
        print('Page Size: ', data['response']['pageSize'])
        print('Number of Pages (Adapt while-loop): ', data['response']['pages'])
    list_result = data['response']['results']
    z_page += 1
    for dics in list_result:
        collector_list.append([dics["webPublicationDate"], BeautifulSoup(dics["fields"]["headline"], "html.parser").get_text(),
                            BeautifulSoup(dics["fields"]["trailText"], "html.parser").get_text(),
                            BeautifulSoup(dics["fields"]["body"], "html.parser").get_text()])

collector_list = np.array(collector_list, dtype=object)
np.save(section + '_2018_2020_object', collector_list)
print('Download finished')
