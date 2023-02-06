"""
Python Code for Lab1
"""
import os
import wget
import zipfile


# set up the working directory
os.chdir("docs/math/causal/data/lab1")

def download_sqf():
    for year in range(2003, 2015):
        url = f"https://www1.nyc.gov/assets/nypd/downloads/zip/analysis_and_planning/stop-question-frisk/sqf-{year}-csv.zip"
        wget.download(url)
        
def download_sqf2():
    for year in range(2015, 2017):
        url = f"https://www1.nyc.gov/assets/nypd/downloads/excel/analysis_and_planning/stop-question-frisk/sqf-{year}.csv"
        wget.download(url)
        
    for year in range(2017, 2019):
        url = f"https://www1.nyc.gov/assets/nypd/downloads/excel/analysis_and_planning/stop-question-frisk/sqf-{year}.xlsx"
        wget.download(url)
    
        
def unzip_file():
    for year in range(2003, 2015):
        with zipfile.ZipFile(f"sqf-{year}-csv.zip", 'r') as zip_ref:
            zip_ref.extractall()
            os.rename(f"{year}.csv", f"sqf-{year}.csv")
            os.remove(f"sqf-{year}-csv.zip")
    
    
if __name__ == "__main__":
    # download and unzip the file
    unzip_file()