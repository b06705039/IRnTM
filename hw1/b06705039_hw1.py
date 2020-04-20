import sys
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords

#read file
filename = sys.argv[1]

#token
f = open(filename, 'r')
tokens = nltk.word_tokenize(f.read())
punctuation = ['\'s','\'']
tokens = [w for w in tokens if w not in string.punctuation and w not in punctuation]

#lowercase
lowercase = [lowercase.lower() for lowercase in tokens]
#stemming
ps = PorterStemmer() 

pss = []
for w in lowercase: 
    pss.append(ps.stem(w))

#stopword
stop_words = set(stopwords.words('english')) 



for r in pss: 
    if not r in stop_words: 
        appendFile = open('result.txt','a') 
        appendFile.write(r+"\n") 
        appendFile.close()
