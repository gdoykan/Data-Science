from os import listdir
from os.path import isfile, join
from collections import Counter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

path = '/Users/pratyushsingh/Data Science Lab EE379K/Lab3/'
files = [f for f in listdir(path) if isfile(join(path, f)) and f.find(".pdf") != -1]
wordCount = {}
words = []

def processWords(words):
    counter = Counter()
    for idx, word in enumerate(words):
        word = word.rstrip()
        word = word.lstrip()
        counter[word] += 1

    return counter

for file in files:
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(file, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    try:
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,caching=caching, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()
        text = text.replace('\n', ' ')
        text = text.split(' ')

        words = words + text
        print("Processed: " + file + " succesfully")

    except:
        print("Error in file: " + file)

    fp.close()
    device.close()
    retstr.close()


from nltk.corpus import words as english_dict
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer

words = [word for word in words if word.isalpha()] #remove punctuation
words = [word.lower() for word in words] #make lowercase
english = set(english_dict.words())
tokens = []

def remove_words():
    for idx, word in enumerate(words):
        if(word in english):
            tokens.append(word)

remove_words() #removing words that are not in the english dictionary


porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens] #stemming the words
freq_distribution_stemmed = FreqDist(stemmed) #frequency distribution of the stems
most_common_stems = freq_distribution_stemmed.most_common(10) #most common stems

freq_distribution_words = FreqDist(tokens)
most_common_words = freq_distribution_words.most_common(10)

print("The ten most common stems are: " + str(most_common_stems))
print("The most common words are: " + str(most_common_words))

#calculating entropy
from math import log

freq = {}
wordsSeen = set()
def calculate_relative_frequency():
    for token in tokens:
        if token in wordsSeen:
            continue
        else:
            freq[token] = freq_distribution_words[token]/len(tokens)
        
def entropy():
    entropy = 0
    for word, frequency in freq.items():
        p_i = frequency
        entropy_of_term = p_i * log(p_i, 2)
        entropy = entropy + entropy_of_term
    
    return entropy * -1

calculate_relative_frequency()
entropy = entropy()
print(entropy)


#synthesizing a paragraph based on the marginal distributions
import numpy as np

keys = list(freq.keys()) #words
frequency = list(freq.values()) #individual frequencies

random_paragraph = np.random.choice(keys, 10000, frequency) #print 1000 words
for word in random_paragraph:
    print(word + " ")

test_freq = FreqDist(random_paragraph) 
print(test_freq.most_common(10))

