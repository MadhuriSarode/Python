# Web scraping

import requests
from bs4 import BeautifulSoup
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk import ngrams
from nltk import word_tokenize, sent_tokenize
nltk.download()

dir_name = "/Users/madhuri/PycharmProjects/Python_Lesson7/"
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".txt"):
        os.remove(os.path.join(dir_name, item))


file_name = 'input.txt'
if not os.path.exists(file_name):  # Creating a file to save the data if it does not exists already
    print("Creating file " + file_name)
    file2 = open(file_name, 'a+', encoding='utf-8')

url = "https://en.wikipedia.org/wiki/Google"
source_code = requests.get(url)  # Make a request and get a web page from the URL
plain_text = source_code.text  # Converting it to plain text
print(plain_text, file=open("input.txt", "a"))  # Saving links in text file
soup = BeautifulSoup(plain_text, 'html.parser')

# Saving the data extracted from web link into the text file titled 'input'
file = open('input1' + '.txt', 'a+', encoding='utf-8')
body = soup.body.get_text()
file.write(body)

with open('input1.txt', 'r', encoding='utf8') as inputData:
    data = inputData.read().replace('\n', '')

# Tokenization
token = nltk.word_tokenize(data)
open('Tokens' + '.txt', 'a+', encoding='utf-8').write(str(token))

sentences = sent_tokenize(data)
open('Sentences' + '.txt', 'a+', encoding='utf-8').write(str(sentences))

# Part Of Speech tagging (POS)
pos = nltk.pos_tag(token)
open('POS' + '.txt', 'a+', encoding='utf-8').write(str(pos))

# Stemming
pStemmer = PorterStemmer()
for a in token:
    with open("PStemmer.txt", "a") as myfile:
        myfile.write(pStemmer.stem(a) + '\n')

lStemmer = LancasterStemmer()
for b in token:
    with open("LStemmer.txt", "a") as myfile:
        myfile.write(lStemmer.stem(b) + '\n')

sStemmer = SnowballStemmer('english')
for c in token:
    with open("SStemmer.txt", "a") as myfile:
        myfile.write(sStemmer.stem(c) + '\n')


# Lemmatization
lemmatizer = WordNetLemmatizer()
for tok in token:
    with open("Lemmatization.txt", "a") as myfile:
        myfile.write(lemmatizer.lemmatize(str(tok)) + '\n')

# Trigram
trigram = ngrams(data.split(), 3)
for gram in trigram:
    with open("Trigram.txt", "a") as myfile:
        myfile.write(str(gram) + '\n')


# Named Entity Recognition
print('Named Entitiy Reoognition is ', ne_chunk(pos_tag(wordpunct_tokenize(data))))
