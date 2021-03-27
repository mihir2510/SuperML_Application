# importing the libraries
from bs4 import BeautifulSoup
import requests
import codecs

import codecs
f=codecs.open("index.html", 'r')
print (f.read())

# Parse the html content
soup = BeautifulSoup(url)
print(soup.prettify())
tds = soup.find_all('div')
print(tds) # print the parsed data of html