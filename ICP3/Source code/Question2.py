# Web scraping

import requests
from bs4 import BeautifulSoup
import os

file_name = 'search_results'
if not os.path.exists(file_name):                               # Creating a file to save the links if it does not exists already
    print("Creating file " + file_name)
    file2 = open(file_name + '.txt', 'a+', encoding='utf-8')

url = "https://en.wikipedia.org/wiki/Deep_learning"
source_code = requests.get(url)                         # Make a request and get a web page from the URL
plain_text = source_code.text                           # Converting it to plain text
soup = BeautifulSoup(plain_text, "html.parser")         # Using beautiful soup to parse HTMl
result_list1 = soup.find('title')                       # Printing out the title of the page
print("The title of the page = ", result_list1.text)

# Save all the links in the file
links_list = []
for link in soup.find_all('a'):             # Iterate over each tag 'a' then save the link in the list variable
    links_list.append(link.get('href'))     # Returning the link using attribute "href" using get

for x in links_list:
    print(x, file=open("search_results.txt", "a"))  # Saving links in text file
