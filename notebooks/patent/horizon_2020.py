# %%
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


def _get_ipc_top(str_list: str) -> list:
    """
    Input: like "['B22F 3/105', 'B29C 67/00']" (a long string wit [])
    Output: ['B22F 3/105', 'B29C 67/00']
    """
    if str_list is np.nan:
        return np.nan
    else:
        str_list = str_list.replace('[', '').replace(']', '')
        str_list = str_list.replace("'", '')
        str_list = str_list.split(",")
        
        ipc_top = [x.strip().split(" ")[0] for x in str_list]
    
    return ipc_top
    

def _is_ai_patent(ipc_list: list) -> int:
    """
    if ipc in in ai_ipc return 1, otherwise 0 
    """
    
    ai_ipc = [
        "G06F", "G06K", "H04N", "G10L",
        "G06T", "A61B", "H04M", "G01N",
        "H04R", "G06N", "G01S", "H04L", "G06Q",
        "H04B", "G09G", "G02B", "G11B", "G08B", "G01B"
    ]
    is_ai = 0
    if ipc_list is np.nan:
        return is_ai
    else: 
        for x in ipc_list:
            if x in ai_ipc:
                is_ai = 1
                break
    
    return is_ai
        

if __name__ == "__main__":
    print("Current working directory:", os.getcwd(), '\n')
    df = pd.read_csv("./pyrio/mdf.csv")
    df.head()
    
    df['ipcTop'] = df['ipc'].apply(lambda x: _get_ipc_top(x))
    
    print(
        df[['ipc', 'ipcTop']].head()
        )
    
    df['aiPatent'] = df['ipcTop'].apply(lambda x: _is_ai_patent(x))
    
    print(
        df[['ipc', 'ipcTop', 'aiPatent']].head()
        )
    
    df.to_csv('./pyrio/mdf.csv', index=False)
    
    

# %%
