from bs4 import BeautifulSoup
import requests
from urllib.request import urlparse
import json

# Global variables
headers = """
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Accept-Encoding: gzip, deflate, br
Accept-Language: en-US,en;q=0.9
Cache-Control: max-age=0
Connection: keep-alive
Cookie: __ssds=2; __ssuzjsr2=a9be0cd8e; __uzmbj2=1674140969; __uzmaj2=bc092231-9990-40c4-9dc9-a2522cc3d5f3; __uzmdj2=1674381382; __uzmcj2=541805878521
Host: data.epo.org
Referer: https://data.epo.org/linked-data/data/publication/EP/2854191/B1/-
Sec-Fetch-Dest: document
Sec-Fetch-Mode: navigate
Sec-Fetch-Site: same-origin
Sec-Fetch-User: ?1
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.188 Safari/537.36 CrKey/1.54.250320
"""

headers = headers.strip().split('\n')
# save it as dict
HEADERS = {x.split(':')[0].strip():
           ("".join(x.split(':')[1:])).strip().replace('//',
                                                       "://") for x in headers}


url = "https://data.epo.org/linked-data/data/publication/EP/2887409/B1/-.json"

page = requests.get(url, headers=HEADERS)
page.encoding = "utf-8"

print(page.status_code)

page_json = page.json()

ipc_info = page_json['result']['primaryTopic']['classificationIPCInventive']

print(type(ipc_info))

print(ipc_info)