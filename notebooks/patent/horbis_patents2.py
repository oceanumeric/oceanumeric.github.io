import os
import re
import math
import json
import time
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from base64 import b64encode



# Global variables
headers = """
Accept: application/json
Accept-Encoding: gzip
Accept-Language: en-US
Authorization:
Host: ops.epo.org
sec-ch-ua: Not_A Brand";v="99
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: macOS
Sec-Fetch-Dest: empty
Sec-Fetch-Mode: cors
Sec-Fetch-Site: cross-site
User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML
X-Forwarded-For: 141.2.233.113
X-Forwarded-Port: 443
X-Forwarded-Proto: https
Connection: keep-alive
"""

headers = headers.strip().split('\n')
# save it as dict
HEADERS = {x.split(':')[0].strip():
           ("".join(x.split(':')[1:])).strip().replace('//',
                                                       "://") for x in headers}



def _refresh_token():
    """
    Refresh the token after 20 minutes 
    """
    auth_url = "https://ops.epo.org/3.2/auth/accesstoken"
    epo_key = 'kwAVrz57UtV9J2ADbPclFGZ9FoWY3Tu8'
    epo_secret = 'cnQYVYjX3ICaax9v'
    epo_auth = (epo_key + ':' + epo_secret).encode('ascii')
    epo_auth = b64encode(epo_auth)
    HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": "Basic {}".format(epo_auth.decode('ascii')),
    }
    payload = {"grant_type": "client_credentials"}
    response = requests.post(auth_url, headers=HEADERS, data=payload)
    print(f"Auth Token Was Refreshed: {response.status_code}", '\n')
    token = response.json()['access_token']
    
    return "Bearer {}".format(token)


def _get_title(title):
    
    title_str = None
    
    if isinstance(title, list):
        for x in title:
            if x['@lang'] == 'en':
                title_str = x['$']
    if isinstance(title, dict):
        if title['@lang'] == 'en':
            title_str = title['$']
                
    return title_str
            

def _get_abstract(abstract_info):
    
    abs_str = None
    
    if isinstance(abstract_info, list):
        for x in abstract_info:
            if x.get('@lang', None) is not None:
                if x.get('@lang') == 'en':
                    p = x.get('p', None)
                    if p is not None:
                        p_content = p.get('$')
                        abs_str = p_content
           
    if isinstance(abstract_info, dict):
        x = abstract_info
        if x.get('@lang', None) is not None:
                if x.get('@lang') == 'en':
                    p = x.get('p', None)
                    if p is not None:
                        p_content = p.get('$')
                        abs_str = p_content
    
    return abs_str


def _get_application_references(applications: dict):
    """
    Return appln_date, appln_num_docdb, appln_num_epodoc
    """
    
    appln_num_epodoc = None
    appln_num_docdb = None
    appln_date = None
    
    if applications is None:
        return appln_date, appln_num_docdb, appln_num_epodoc
    
    docs = applications.get('document-id', None)
    
    if docs is not None:
        if isinstance(docs, list):
            for x in docs:
                if x.get('@document-id-type', None) == 'epodoc':
                    appln_num_epodoc = x.get('doc-number')['$']
                    appln_date = x.get('date')['$']
                if x.get('@document-id-type', None) == 'docdb':
                    ctry = x.get('country')['$']
                    num = x.get('doc-number')['$']
                    appln_num_docdb = ctry + num
                    
        elif isinstance(docs, dict):
            if docs.get('@document-id-type', None) == 'epodoc':
                    appln_num_epodoc = docs.get('doc-number')['$']
                    appln_date = docs.get('date')['$']
            
            if docs.get('@document-id-type', None) == 'docdb':
                    ctry = docs.get('country')['$']
                    num = docs.get('doc-number')['$']
                    appln_num_docdb = ctry + num
    
    return  appln_date, appln_num_docdb, appln_num_epodoc


def _get_priority(priority_claim: dict):
    """
    return pc_date, pc_num   
    """
    
    pc_date = None
    pc_num = None
    
    priority = priority_claim.get('priority-claim', None)
    
    if isinstance(priority, list):
        priority = priority[0]
        priority = priority.get('document-id', None)
        if priority is not None:
            if isinstance(priority, list):
                for x in priority:
                    if x.get('@document-id-type') == 'epodoc':
                        pc_num = x.get('doc-number')['$']
                        pc_date = x.get('date')['$']
                        
    if isinstance(priority, dict):
        priority = priority.get('document-id', None)
        if priority is not None:
            if isinstance(priority, list):
                for x in priority:
                    if x.get('@document-id-type') == 'epodoc':
                        pc_num = x.get('doc-number')['$']
                        pc_date = x.get('date')['$']
        
        
    return pc_date, pc_num    


def _get_cpc(cpc_class: dict):
    
    cpc_list = []
    
    cpc = cpc_class.get('patent-classification', None)
    
    if cpc is not None:
        if isinstance(cpc, list):
            for x in cpc:
                if x['classification-scheme']['@scheme'] == 'CPCI':
                    temp = ""
                    temp += x['section']['$']
                    temp += x['class']['$']
                    temp += x['subclass']['$']
                    temp += x['main-group']['$']
                    temp = temp + '/' + x['subgroup']['$']
                    temp += f"({x['generating-office']['$']})"
                    
                    cpc_list.append(temp)
        if isinstance(cpc, dict):
            x = cpc
            if x['classification-scheme']['@scheme'] == 'CPCI':
                x = cpc
                temp = ""
                temp += x['section']['$']
                temp += x['class']['$']
                temp += x['subclass']['$']
                temp += x['main-group']['$']
                temp = temp + '/' + x['subgroup']['$']
                temp += f"({x['generating-office']['$']})"
                
                cpc_list.append(temp)
            
    return cpc_list

def _get_ipc(class_ipc):
    
    ipc_list = []
    
    ipc = class_ipc.get('classification-ipcr', None)
    
    if ipc is not None:
        if isinstance(ipc, list):
            for x in ipc:
                temp = x['text']['$']
                ipc_list.append(temp.replace(' ', ''))
        if isinstance(ipc, dict):
            x = ipc
            temp = x['text']['$']
            ipc_list.append(temp.replace(' ', ''))
            
    return ipc_list


def _get_pubs(pubs):
    """
    return granted, grantDate, pub_list
    """
    
    pub_list = []
    granted = 0
    grantDate = None
    
    pubs = pubs.get('document-id')
    
    if isinstance(pubs, list):
        for x in pubs:
            if x.get('kind', None) is not None:
                temp = x['country']['$'] + x['doc-number']['$']
                kind = x['kind']['$'] 
                temp = temp + f"({kind})"
                
                pub_list.append(temp)
                
                if 'B' in kind:
                    granted = 1
                    grantDate = x['date']['$']

    if isinstance(pubs, dict):
        x = pubs
        if x.get('kind', None) is not None:
                temp = x['country']['$'] + x['doc-number']['$']
                kind = x['kind']['$'] 
                temp = temp + f"({kind})"
                
                pub_list.append(temp)
                
                if 'B' in kind:
                    granted = 1
                    grantDate = x['date']['$']
                    
    return granted, grantDate, pub_list
        
            
def get_bibliographic_data(patent_num: str, token):
    """
    Get bibliographic data based on patent number
    -----------------------------------------------
    Input: cell from csv file created from R 
    """
    patent_num = str(patent_num)
    
    patent_info = {
        "patentNumber": patent_num,
        "kindCode": None,
        "familyID" : None,
        'granted': None,
        'grantDate': None,
        'applicationDate': None,
        'application_docdb': None, 
        'application_epodoc': None,
        'priorityDate': None,
        'priorityNum': None, 
        'ipcClass': None,
        'cpcClass': None,
        'title': None,
        'abstract': None,
        'publications': None
    }
    

    if patent_num == 'nan':
        patent_info = {"patentNumber": None}
        return patent_info
    
    
    url = "http://ops.epo.org/3.2/rest-services/published-data/"
    url += "publication/epodoc/"
    url = url + patent_num + "/biblio"
    
    HEADERS["Authorization"] = token

    page = requests.get(url=url, headers=HEADERS)
    page.encoding = "utf-8"
    print("Response status: -------", page.status_code, '\n')
    
    if page.status_code == 200:
        page_json = page.json()
        
        # 'ops:world-patent-data'
        data = page_json['ops:world-patent-data']
        document = data["exchange-documents"].get("exchange-document")
        
        if isinstance(document, list):
            
            document = document[0]
            
            fm_id = document.get("@family-id", None)
            patent_info["familyID"] = fm_id
            
            kind_code = document.get("@kind", None)
            patent_info["kindCode"] = kind_code
            
            abstract = document.get("abstract", None)
            patent_info['abstract'] = _get_abstract(abstract)
            
            # get all bibliographic information 
            bib_data = document.get('bibliographic-data', None)
            if bib_data is not None:
                applications = bib_data.get('application-reference', None)
                if applications is not None:
                    appln_date, docdb, epodoc = _get_application_references(
                    applications)
                    patent_info['applicationDate'] = appln_date
                    patent_info['application_docdb'] = docdb
                    patent_info['application_epodoc'] = epodoc
                    
                priority = bib_data.get('priority-claims', None)
                if priority is not None:
                    pd, pn = _get_priority(priority)
                    patent_info['priorityDate'] = pd
                    patent_info['priorityNum'] = pn
                    
                cpc = bib_data.get('patent-classifications', None)
                if cpc is not None:
                    patent_info['cpcClass'] = _get_cpc(cpc)
                    
                ipc = bib_data.get('classifications-ipcr', None)
                if ipc is not None:
                    patent_info['ipcClass'] = _get_ipc(ipc)
                    
                title = bib_data.get('invention-title', None)
                if title is not None:
                    patent_info['title'] = _get_title(title).upper()
                    
                publications = bib_data.get('publication-reference', None)
                if publications is not None:
                    gt, gtdt, gtlist = _get_pubs(publications)
                    patent_info['granted'] = gt
                    patent_info['grantDate'] = gtdt
                    patent_info['publications'] = gtlist
                    
        else:
            
            fm_id = document.get("@family-id", None)
            patent_info["familyID"] = fm_id
            
            kind_code = document.get("@kind", None)
            patent_info["kindCode"] = kind_code
            
            abstract = document.get("abstract", None)
            patent_info['abstract'] = _get_abstract(abstract)
            
            # get all bibliographic information 
            bib_data = document.get('bibliographic-data', None)
            if bib_data is not None:
                applications = bib_data.get('application-reference', None)
                if applications is not None:
                    appln_date, docdb, epodoc = _get_application_references(
                    applications)
                    patent_info['applicationDate'] = appln_date
                    patent_info['application_docdb'] = docdb
                    patent_info['application_epodoc'] = epodoc
                
                priority = bib_data.get('priority-claims', None)
                if priority is not None:
                    pd, pn = _get_priority(priority)
                    patent_info['priorityDate'] = pd
                    patent_info['priorityNum'] = pn
                    
                cpc = bib_data.get('patent-classifications', None)
                if cpc is not None:
                    patent_info['cpcClass'] = _get_cpc(cpc)
                    
                ipc = bib_data.get('classifications-ipcr', None)
                if ipc is not None:
                    patent_info['ipcClass'] = _get_ipc(ipc)
                    
                title = bib_data.get('invention-title', None)
                if title is not None:
                    patent_info['title'] = _get_title(title).upper()
                    
                publications = bib_data.get('publication-reference', None)
                if publications is not None:
                    gt, gtdt, gtlist = _get_pubs(publications)
                    patent_info['granted'] = gt
                    patent_info['grantDate'] = gtdt
                    patent_info['publications'] = gtlist
        
        return patent_info
    else:
        if "WO2000" in patent_num:
            patent_num = patent_num.replace('WO200', 'WO')
            return get_bibliographic_data(patent_num, token)
        elif 'WO20' in patent_num:
            patent_num = patent_num.replace('WO20', 'WO')
            patent_num = patent_num.replace('0', '')
            return get_bibliographic_data(patent_num, token)
        elif 'WO19' in patent_num:
            patent_num = patent_num.replace('WO19', 'WO')
            patent_num = patent_num.replace('0', '')
            return get_bibliographic_data(patent_num, token)
        else:
            print("------- !**! --- Patent Not found for", patent_num, '\n')
            return patent_info
    
        
    
def unit_test1():
    """
    based on horbis_de_patents.csv
    """
    print("::::::::::::::::::::::Unit Test Running::::::::::::::::::::::\n")
    horbis_patents = pd.read_csv('./data/horbis_de_patents.csv')
    # horbis_patents_wo = horbis_patents[horbis_patents['Publn_auth'] == 'US']
    test_sample = horbis_patents.sample(5)
    print(
        test_sample, '\n'
    )
    
    token = _refresh_token()
    
    # initialize df for results 
    patents = pd.DataFrame()
    
    for i, idx in enumerate(test_sample.index):
        time.sleep(0.2)
        
        patent_number = test_sample['Patent_number'][idx]
        print(f"GETTING data for -----------------:{i}-{idx}: {patent_number}\n")
        patent_number = horbis_patents['Patent_number'][idx]
        print(
            get_bibliographic_data(patent_number, token), '\n'
        )


def unit_test2():
    """
    based on horbis_de_patents.csv
    """
    print("::::::::::::::::::::::Unit Test Running::::::::::::::::::::::\n")
    horbis_patents = pd.read_csv('./data/horbis_de_patents.csv')
    # horbis_patents_wo = horbis_patents[horbis_patents['Publn_auth'] == 'US']
    start = time.time()
    token = _refresh_token()
    
    # initialize df for results 
    epo_df = pd.DataFrame()
    
    for i, idx in enumerate(horbis_patents.index):
        time.sleep(0.3)
        now = time.time()
        lapse = now - start

        if lapse >= 300:
            token = _refresh_token()
            time.sleep(60)
            print("----------- It has been 5 MINUTES ------ :)")
            start = time.time()
            
            
        patent_number = horbis_patents['Patent_number'][idx]
        print(f"GETTING data for -------------: {i} -{idx}: {patent_number}\n")
        patent_number = horbis_patents['Patent_number'][idx]
        pt_dict = get_bibliographic_data(patent_number, token)
        
        # append dict into df 
        temp = pd.DataFrame.from_dict(
                pt_dict, orient='index'
            ).transpose()
        epo_df = pd.concat([epo_df, temp], ignore_index=True)
        epo_df.to_csv('./data/horbis_de_epo.csv', index=False)
        
        
def get_89626_179252():
    """
    based on horbis_de_patents.csv
    """
    print("::::::::::::::::::::::Unit Test Running::::::::::::::::::::::\n")
    horbis_patents = pd.read_csv('./data/horbis_de_patents.csv')
    horbis_patents = horbis_patents.iloc[89626:179252, :]
    start = time.time()
    token = _refresh_token()
    
    # initialize df for results 
    epo_df = pd.DataFrame()
    
    for i, idx in enumerate(horbis_patents.index):
        time.sleep(np.random.uniform(0.2, 0.5))
        now = time.time()
        lapse = now - start

        if lapse >= 300:
            token = _refresh_token()
            time.sleep(30)
            print("----------- It has been 5 MINUTES ------ :)")
            start = time.time()
            
            
        patent_number = horbis_patents['Patent_number'][idx]
        print(f"GETTING data for -------------: {i} -{idx}: {patent_number}\n")
        patent_number = horbis_patents['Patent_number'][idx]
        pt_dict = get_bibliographic_data(patent_number, token)
        
        # append dict into df 
        temp = pd.DataFrame.from_dict(
                pt_dict, orient='index'
            ).transpose()
        epo_df = pd.concat([epo_df, temp], ignore_index=True)
        epo_df.to_csv('./data/horbis_de_epo7.csv', index=False)
    
    
if __name__ == "__main__":
    print("Current working directory:", os.getcwd(), '\n')
 
    # test 1
    get_89626_179252()
    # token = 'Bearer DDX6TaFo602n5paqjh40w4LN1GIJ'
    # print(
    # get_bibliographic_data("US6239155", token)
    # )
    
    
    