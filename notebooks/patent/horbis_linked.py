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
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7
Accept-Encoding: gzip, deflate, br
Accept-Language: en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7
Cache-Control: max-age=0
Connection: keep-alive
Cookie: __ssds=2; __ssuzjsr2=a9be0cd8e; __uzmbj2=1673964576; __uzmdj2=1673964576; __uzmaj2=286bb21d-b9ca-4d83-9aa7-405136bf4072; __uzmcj2=370911014280
Host: data.epo.org
sec-ch-ua: "Chromium";v="110", "Not A(Brand";v="24", "Microsoft Edge";v="110"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
Sec-Fetch-Dest: document
Sec-Fetch-Mode: navigate
Sec-Fetch-Site: none
Sec-Fetch-User: ?1
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.50
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
    pub_url = url + authority + '/' + patent_number + '.json'
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
        - doc_link 
    """
    pub_slag = 'publication/'
    pub_idx = pub_url.find(pub_slag) + len(pub_slag)
    authority = pub_url[pub_idx:pub_idx + 2]
    page = requests.get(pub_url, headers=HEADERS)
    page.encoding = "utf-8"
    print("Response status: -------", page.status_code)
    page_json = page.json()

    items = page_json['result'].get('items', None)

    if items is not None and isinstance(items, list):
        if len(items) > 0:
            doc_link = items[0].get('_about', None)
            application_number = items[0].get('application', None)
            if application_number is not None:
                application_number = application_number.get('applicationNumber')
                return authority + application_number, doc_link
            else:
                return None, None
        else:
            return None, None
    elif items is not None and isinstance(items, dict):
        doc_link = items.get('_about', None)
        application_number = items.get('application', None)
        if application_number is not None:
            application_number = application_number.get('applicationNumber')
            return authority + application_number, doc_link
        else:
            return None, None
    else:
        return None, None
        


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
    application_url = url + authority + '/' + application_number + '.json'
    return application_url


def get_application_info(application_url: str) -> dict:
    """
    Get key application information based on application_url

    ------------
    Input: application_url, such as
    https://data.epo.org/linked-data/doc/application/US/201314434959.json

    Output: a dictionary
    """
    application_info = {
        'familyID': None,
        'applicationDate': None,
        'granted': 0,
        'grantDate': None,
        'cpcClass': None,
        'publicationItems': None
    }

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
    if family is not None and isinstance(family, dict):
        family_url = family['_about']
        temp_slag = 'simple-family/'
        temp_idx = family_url.find(temp_slag) + len(temp_slag)
        application_info['familyID'] = family_url[temp_idx:]
    elif isinstance(family, str):
        family_url = family
        temp_slag = 'simple-family/'
        temp_idx = family_url.find(temp_slag) + len(temp_slag)
        application_info['familyID'] = family_url[temp_idx:]
    else:
        application_info['familyID'] = family


    cpc = result.get("classificationCPCInventive", None)

    if cpc is not None:
        if isinstance(cpc, list):
            application_info['cpcClass'] = [
                x['label'] for x in cpc if isinstance(x, dict)
                ]
        elif isinstance(cpc, dict):
            application_info['cpcClass'] = cpc.get('label', None)
        else:
            application_info['cpcClass'] = cpc
    else:
        application_info['cpcClass'] = cpc

    # publications
    publication = result.get('publication', None)

    if publication is not None:
        if isinstance(publication, list):
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
        if isinstance(publication_items, str):
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


def _construct_pub_kind_url(pub_kind_b_number: str) -> str:
    """
    Input: - 'EP 2245642 B1'

    Output:
    https://data.epo.org/linked-data/data/publication/EP/2245642/B1/-.json
    """
    b_kind_list = pub_kind_b_number.split(' ')
    base_url = 'https://data.epo.org/linked-data/data/publication/'
    for x in b_kind_list:
        base_url = base_url + x + '/'
    url = base_url + '-.json'

    return url


def get_epo_publication_info(pub_kind_url: str) -> dict:
    """
    Get EPO's publication information based on pub_url

    --------------------------
    Input:
        - pub_kind_url such as
        https://data.epo.org/linked-data/data/publication/EP/3438695/B1/-.json
    Output:
        - a dictionary
    """
    pub_info = {
        "publicationDate": None,
        "priorityNumber": None,
        "language": None,
        "ipc": None,
        "title": None,
        "abstract": None
    }

    page = requests.get(pub_kind_url, headers=HEADERS)
    page.encoding = "utf-8"
    print("Response status: -------", page.status_code)
    page_json = page.json()

    result = page_json['result']["primaryTopic"]

    pub_info['publicationDate'] = result.get("publicationDate", None)

    # priority
    priority = result.get("priority", None)
    if priority is not None:
        if isinstance(priority, list):
            priority_num = priority[0].replace(
                'http://data.epo.org/linked-data/id/application/', ''
            )
        else:
            priority_num = priority.replace(
                'http://data.epo.org/linked-data/id/application/', ''
            )
        pub_info['priorityNumber'] = priority_num
    else:
        pub_info['priorityNumber'] = priority

    language = result.get("language", None)
    if language is not None:
        pub_info['language'] = language[-2:]
    else:
        pub_info['language'] = None

    pub_info['ipc'] = []
    ipc = result.get("classificationIPCInventive", None)
    if ipc is not None:
        if isinstance(ipc, list):
            for x in ipc:
                if isinstance(x, dict):
                    pub_info['ipc'].append(x['label'])
                else:
                    ipc_label = x.replace(
                        'http://data.epo.org/linked-data/def/ipc/',
                        ''
                    )
                    pub_info['ipc'].append(ipc_label)
        elif isinstance(ipc, dict):
            pub_info['ipc'] = ipc.get('label', None)
        else:
            pub_info['ipc'] = ipc
    else:
        pub_info['ipc'] = ipc

    title = result.get("titleOfInvention", None)
    if title is not None and isinstance(title, list):
        pub_info['title'] = title[0]
    elif isinstance(title, str):
        pub_info['title'] = title[0]
    else:
        pub_info['title'] = title

    abstract = result.get("abstract",  None) 
    if abstract is not None:
        pub_info['abstract'] = abstract

    return pub_info


def extract_epo_linked_data(patent_number: str) -> dict:
    """
    Extract patent information from EPO Linked data based on
    patent number
    --------------------
    Return a dictionary
    """

    patent_num = str(patent_number)
    
    patent_info = {
        "patentNumber": patent_num,
        "applicationNum": None, 
    }

    pub_url = construct_pub_url(patent_num)

    appln_num, doc_link = get_application_number(pub_url)
    patent_info['applicationNum'] = appln_num
    
    if appln_num is None:
        patent_info = {
            'patentNumber': 'EP0925591', 
            'applicationNum': None, 
            'familyID': None, 
            'applicationDate': None, 
            'granted': None, 
            'grantDate': None, 
            'cpcClass': None, 
            'publicationItems': None, 
            'publicationDate': None, 
            'priorityNumber': None, 
            'language': None, 
            'ipc': None, 
            'title': None, 
            'abstract': None}
        return patent_info

    time.sleep(0.1)

    appln_url = construct_application_url(appln_num)
    appln_dict = get_application_info(appln_url)
    patent_info = patent_info | appln_dict

    time.sleep(0.1)
    doc_link = doc_link + '.json'
    pub_dict = get_epo_publication_info(doc_link)
    patent_info = patent_info | pub_dict
    
    return patent_info


def unit_test():
    """
    based on horbis_de_patents.csv
    """
    print("::::::::::::::::::::::Unit Test Running::::::::::::::::::::::\n")
    horbis_patents = pd.read_csv('./data/horbis_de_patents.csv')
    horbis_patents = horbis_patents[horbis_patents['Publn_auth'] == 'EP']
    horbis_patents = horbis_patents.iloc[63443:, :]
    
    
    start = time.time()
    
    # initialize df for results 
    epo_df = pd.DataFrame()
    
    for i, idx in enumerate(horbis_patents.index):
        time.sleep(0.2)
        now = time.time()
        lapse = now - start

        if lapse >= 300:
            time.sleep(20)
            print("----------- It has been 5 MINUTES ------ :)")
            start = time.time()
            
            
        patent_number = horbis_patents['Patent_number'][idx]
        print(f"GETTING data for -------------: {i} -{idx}: {patent_number}\n")

        pt_dict = extract_epo_linked_data(patent_number)
        
        # append dict into df 
        temp = pd.DataFrame.from_dict(
                pt_dict, orient='index'
            ).transpose()
        epo_df = pd.concat([epo_df, temp], ignore_index=True)
        epo_df.to_csv('./epodata/horbis_de_linked3.csv', index=False)

if __name__ == "__main__":
    print("Current working directory:", os.getcwd(), '\n')
    unit_test()
    # print(
    # extract_epo_linked_data('EP3024617')
    # )
    
