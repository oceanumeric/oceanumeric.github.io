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


def get_application_number(pub_url: str) -> str:
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
    
    if items is not None and len(items) != 0:
        application_number = items[0].get('application', None)
        if application_number is not None:
            application_number = application_number.get('applicationNumber')
            return authority + application_number
        else:
            return None
    else:
        return None
    

def construct_application_url(application_number: str) -> str:
    """
    Construct publication url based on the application_number, like EP17198906
    or US10059425
    ----------
    Input:
        - application in format {EP}+{Number}
    
    Output: 
        - application url application/{authority}{number}.json 
        - authority 
    """
    authority = application_number[:2]
    application_number = application_number[2:]
    url = 'https://data.epo.org/linked-data/doc/application/'
    application_url = url+authority + '/' + application_number+'.json' 
    return application_url


def get_application_info(application_url: str) -> dict:
    """
    Get key application information based on application_url
    
    ------------
    Input: application_url, such as 
    https://data.epo.org/linked-data/doc/application/US/201314434959.json
    
    Output: a dictionary
    """
    application_info = {}
    
    page = requests.get(application_url, headers=HEADERS)
    page.encoding = "utf-8"
    print("Response status: -------", page.status_code)
    page_json = page.json()
    
    result = page_json['result']["primaryTopic"]
    
    application_info['applicationDate'] = result['filingDate']
    
    
    grant = result.get("grantDate", None)
    
    if grant is None:
        application_info['granted'] = 0
    else:
        application_info['granted'] = 1
        
    application_info["grantDate"] = grant
    
    family = result.get("family", None)
    if family is not None:
        family_url = family['_about']
        temp_slag = 'simple-family/'
        temp_idx = family_url.find(temp_slag) + len(temp_slag)
        application_info['familyID'] = family_url[temp_idx:]
    
    cpc = result.get("classificationCPCInventive", None)
    
    if cpc is not None:
        if type(cpc) is list:
            application_info['cpcTags'] = [x['label'] for x in cpc]
        elif type(cpc) is dict:
            application_info['cpcTags'] = cpc.get('label', None)
        else:
            application_info['cpcTags'] = cpc          
    else:
        application_info['cpcTags'] = cpc
    
    # publications
    publication = result.get('publication', None)
    
    if publication is not None:
        if type(publication) is list:
            application_info['publicationItems'] = [
                x['label'] for x in publication
                ]
        else:
            application_info['publicationItems'] = publication['label']
    else:
        application_info['publicationItems'] = None
    
        
    return application_info


def _get_b_doc(publication_items) -> str:
    """
    Extract publication number with 'B' kind code
    
    -----------------
    
    Input:
        - publication_items: list
        - publication_items: str
        - publication_items: None
        
    Output:
        - str like 'EP 3533599 A1'
    """
    if publication_items is not None:
        if type(publication_items) is str:
            temp = publication_items.split(' ')
            if 'B' in temp[-1]:
                return publication_items
            else:
                return None
        else:
            for x in publication_items:
                temp = x.split(' ')
                if 'B1' in temp:
                    return x
                elif 'B' in temp[-1]:
                    return x
            return None
                

def get_publication_info(pub_kind_url: str) -> dict:
    """
    Get publication information based on pub_url
    
    --------------------------
    Input:
        - pub_kind_url such as 
        https://data.epo.org/linked-data/data/publication/EP/3438695/B1/-.json
    Output:
        - a dictionary
    """
    pub_info = {}
    
    page = requests.get(pub_kind_url, headers=HEADERS)
    page.encoding = "utf-8"
    print("Response status: -------", page.status_code)
    page_json = page.json()
    
    result = page_json['result']["primaryTopic"]
    
    pub_info['publicationDate'] = result.get("publicationDate", None)
    
    language = result.get("language", None)
    if language is not None:
        pub_info['language'] = language[-2:]
    else:
        pub_info['language'] = None    
        
    ipc = result.get("classificationIPCAdditional", None)
    if ipc is not None:
        if type(ipc) is list:
            pub_info['ipc'] = [x['label'] for x in ipc]
        elif type(ipc) is dict:
            pub_info['ipc'] = ipc.get('label', None)
        else:
            pub_info['ipc'] = ipc
    else:
        pub_info['ipc'] = ipc
    
    return pub_info


if __name__ == "__main__":
    print("Current working directory:", os.getcwd(), '\n')
    print("::::::::::::::::::::::Unit Test Running::::::::::::::::::::::\n")
    airbus_han_patents = pd.read_csv('./data/airbus_han_patents.csv')
    test_sample = airbus_han_patents.sample(2)
    print(
        test_sample, '\n'
    )
    foo_df = pd.DataFrame()
    for idx in test_sample.index:
        print("************ Unit Test:", idx)
        patent_number = test_sample['Patent_number'][idx]
        pub_url = construct_pub_url(patent_number)
        print("Testing construct_pub_url:", pub_url)
        application_number = get_application_number(pub_url)
        print("Testing get_application_number:", application_number)
        if application_number is not None:
            application_url = construct_application_url(application_number)
            print("Testing construct_application_url:", application_url)
            application_info = get_application_info(application_url)
            print("Testing get_application_info:", application_info)
            temp = pd.DataFrame.from_dict(
                    application_info, orient='index'
                    ).transpose()
            foo_df = pd.concat([foo_df, temp], ignore_index=True)
            b_doc = _get_b_doc(application_info['publicationItems'])
            print("Testing _get_b_doc:", b_doc)
        else:
            print("NO application number was found for", patent_number)
        # print("Testing get_pub_info:", get_publication_info())
        
        print(foo_df)
        print('\n')
# %%
