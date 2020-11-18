from urllib import request
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import os

# code partialy from stackoverwlow https://stackoverflow.com/questions/54616638/download-all-pdf-files-from-a-website-using-python

# connect to website and get list of all pdfs
url = "https://www.precisionfloorplan.com/floorplan-database/"

folder_location = r'precision_floorplans'
if not os.path.exists(folder_location):os.mkdir(folder_location)
response = request.urlopen(url).read()
soup = BeautifulSoup(response, "html.parser")
for link in soup.findAll('a', attrs={'href': re.compile("address")}):
    url2 = "https://www.precisionfloorplan.com/floorplan-database/" + link.get('href')
    response2 = request.urlopen(url2).read()
    soup2 = BeautifulSoup(response2, "html.parser")
    for link2 in soup2.select("a[href$='.pdf']"):
        filename = os.path.join(folder_location, link2['href'].split('/')[-1])
        print(filename)

        with open(filename, 'wb') as f:
            f.write(requests.get(urljoin(url2, link2['href'])).content)