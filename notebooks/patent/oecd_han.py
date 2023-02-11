import os
import json
import time
import requests
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# Global variables
headers = """
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Host: data.epo.org
User-Agent: Chrome/88.0.4324.188 
"""

headers = headers.strip().split('\n')
# save it as dict
HEADERS = {x.split(':')[0].strip():
           ("".join(x.split(':')[1:])).strip().replace('//',
                                                       "://") for x in headers}


def url_to_ipc(url):
    """
    request ipc classification based on EPO's Open Linked data API
    """
    ipcTags = []
    page = requests.get(url, headers=HEADERS)
    page.encoding = "utf-8"
    time.sleep(0.3)
    print("Response status: ------- ", page.status_code)
    page_json = page.json()
    ipc_info = page_json['result']['primaryTopic'].get(
        'classificationIPCInventive', None
    )
    
    if ipc_info is not None:
        type_check = type(ipc_info)
        if type_check is list:
            for x in ipc_info:
                x = x.replace('http://data.epo.org/linked-data/def/ipc/', '')
                ipcTags.append(x)
        elif type_check is dict:
            ipcTags.append(x.get('label', None))
        elif type_check is str:
            ipc_info = ipc_info.replace(
                'http://data.epo.org/linked-data/def/ipc/', ''
                )
            ipcTags.append(ipc_info)
    
    return ipcTags

    
def construct_url(patent_number):
    patent_number = patent_number.replace("EP", "")
    url = 'https://data.epo.org/linked-data/data/publication/EP/'
    url = url+patent_number+'/B1/-.json?_view=description' 
    
    return url


def pn_to_pic(patent_number):
    url = construct_url(patent_number)
    ipc_list = url_to_ipc(url)
    return ipc_list


def ipc_collection(ipc_sum):
    """
    collect all ipc tags
    --------------------
    Input: pandas df column sum such as df['ipc'].sum()
    Output: a list of all ipc tags
    """
    ipc_sum = ipc_sum.replace("'", "")
    ipc_sum = ipc_sum[1:].replace('[', '')
    ipc_sum = ipc_sum.replace(']', ",")
    ipc_list = ipc_sum.split(',')
    ipc_list = ipc_list[:-1]
    
    ipc_list = [x.strip() for x in ipc_list]
    
    return ipc_list
    

def get_top_ipc(ipc_list):
    """
    extract top ipc class for a ipc_list
    """
    top_ipc = [x[0] for x in ipc_list]
    top_ipc_sub = [x[:4] for x in ipc_list]
    
    return top_ipc, top_ipc_sub
    

if __name__ == "__main__":
    # -------- test key functions -------- 
    # pn = 'EP2837556'
    # purl = construct_url(pn)
    # print(url_to_ipc(purl))
    # ----------- read csv and process ------------
    # airbus = pd.read_csv('notebooks/patent/data/airbus_granted.csv')
    # # filter out those granted
    # airbus = airbus[airbus['granted'] == 1]
    # airbus['ipc'] = airbus['Patent_number'].apply(lambda x: pn_to_pic(x))
    # airbus.to_csv('notebooks/patent/data/airbus_granted_ipc.csv', index=False)
    # ------------ generate summary statistics ------------