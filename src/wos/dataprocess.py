import os
import xlrd
import pandas as pd


def combine_record_citation():
    '''
    A function to combine two excel files:
        - record.xlsx includes author, title, abstract, etc. 
        - citation.xlsx includes backward citation 
    '''
    innobem = pd.DataFrame()
    path = '/Users/Michael/Downloads/wos2/'
    for i in range(1, 11):
        record = 'r'+str(i)+'.xls'
        citation = 'c'+str(i)+'.xls'
        # read file
        record = path + record 
        citation = path + citation
        df1 = pd.read_excel(record)
        df2 = pd.read_excel(citation)
        # select column names 
        columns1 = ['UT (Unique WOS ID)', 'Authors','Author Full Names',
                    'Article Title', 'Source Title','Author Keywords', 
                    'Keywords Plus', 'Abstract', 'Addresses', 'Affiliations',
                    'Publication Year', 'DOI']
        columns2 = ['Cited References', 'Cited Reference Count']
        # merge dataset
        df1 = df1[columns1]
        df2 = df2[columns2]
        df_combine = pd.concat([df1, df2], ignore_index=True, axis=1)
        innobem = pd.concat([innobem, df_combine], ignore_index=True)
    
    innobem.columns = ['wos_id', 'authors', 'full_names', 'title', 'journal',
                       'keywords', 'extra_keywords', 'abstract', 'address',
                       'affiliation', 'year', 'doi', 'cited_references', 
                       'cited_references_count',]
    
    innobem.to_csv('/Users/Michael/Downloads/wos2/wos2_dataset.csv', index=False)
    

        
if __name__ == "__main__":
    print(os.getcwd())
    combine_record_citation()