# %%
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


def construct_pub_url(patent_number: str) -> str:
    """
    Construct publication url based on the patent_number, like EP3477328
    or US10059425
    ----------
    Input:
        - patent number in format {EP}+{Number}
    
    Output: 
        - publication url publication/{authority}{number}.json 
        - authority 
    """
    authority = patent_number[:2]
    patent_number = patent_number[2:]
    url = 'https://data.epo.org/linked-data/data/publication/'
    pub_url = url+authority + '/' + patent_number+'.json' 
    return pub_url


def _get_application_number(pub_url: str) -> str:
    """
    Get application number based on publication url such as
    https://data.epo.org/linked-data/data/publication/EP/3291094.json
    -------------
    Input: 
        - pub_url
    
    Output: 
        - application_number
    """
    pub_slag = 'publication/'
    pub_idx = pub_url.find(pub_slag) + len(pub_slag)
    authority = pub_url[pub_idx:pub_idx+2]
    page = requests.get(pub_url, headers=HEADERS)
    page.encoding = "utf-8"
    print("Response status: -------", page.status_code)
    page_json = page.json()
    
    items = page_json['result'].get('items', None)
    
    if items is not None:
        application_number = items[0]['application']['applicationNumber']
        return authority + application_number
    else:
        return None
    

def _get_application_info(application_number: str):
    """
    Get key application information based on application number
    such as EP17198906
    ------------
    Input: application_number
    Output: 
    """




if __name__ == "__main__":
    print(os.getcwd())
    airbus_han_patents = pd.read_csv('./data/airbus_han_patents.csv')
    pub_url = construct_pub_url('EP3477328')
    print(
        _get_application_number(pub_url)
    )
# %%
