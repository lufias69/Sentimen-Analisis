import pandas as pd
import time
import numpy as np
import re
import pickle
import string, unicodedata
from unidecode import unidecode

data_replace  = pd.read_excel('data/Corpus_kata_replace_new.xlsx', sheet_name = 'kata_repalce')
data_hapus  = pd.read_excel('data/Corpus_kata_replace_new.xlsx', sheet_name = 'kata_hapus')

diganti = data_replace['kata'].tolist()
ganti = data_replace['ganti'].tolist()
hapus = data_hapus['kata'].tolist()


#fungsi fungsi ini digunakan untuk implementasi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#factory = StemmerFactory()
#factoryStop = StopWordRemoverFactory()
#stemmer = factory.create_stemmer()
#stopword = factoryStop.create_stop_word_remover()
factory = StemmerFactory()
factoryStop = StopWordRemoverFactory()
stemmer = factory.create_stemmer()
stopword = factoryStop.create_stop_word_remover()



def stemmer_stopWord(str):
    teks = stemmer.stem(str)
    teks = stopword.remove(teks)
    return teks



def replace_word(teks):
    teks_tokenize = teks.split()
    for j in range(len(teks_tokenize)):
        for k in range(len(diganti)):
            if teks_tokenize[j]==diganti[k]:
                teks_tokenize[j] = ganti[k]
        for k in range(len(hapus)):
            if teks_tokenize[j] == hapus[k]:
                teks_tokenize[j]= ''
    join = ' '.join(map(str,(teks_tokenize)))
    join = re.sub('[\s]+', ' ', join)
    return join

def removePunc(str):
    str = re.sub(r'[^\w]|_',' ',str)
    str = re.sub(r"\b\d+\b", " ", str)
    str = re.sub('[\s]+', ' ', str)
    return str
#x = removePunc(replace_sw(teks))


fo = pd.read_excel('data/karakter.xlsx', sheet_name='Sheet1')
x = fo['karakter'].tolist() #
y = fo['replace'].tolist()

def gantiKarakter(str, x=x, y=y):
    for i in range(len(x)):
        if i == 0:
            n_word = str
        n_word = n_word.replace(x[i],y[i])
    return unidecode(n_word).lower()

def normalAt(str):
    ok = gantiKarakter(str)
    n_w = []
    for i in range(len(ok)):
        if ok[i] == "@" and i !=0 and ok[i-1] !=" ":
            n_w.append(" @")
        else:
            n_w.append(ok[i])
    return "".join(n_w)

def cleaning(str):
    #remove non-ascii
    str = unicodedata.normalize('NFKD', str).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    #remove URLs
    str = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', str)
    str = str.lower()
    #Remove additional white spaces
    str = re.sub('[\s]+', ' ', str)
       
    return str
def getJtext(text):
    words = re.findall(r'[a-z0-9@.]+', text)
    return ' '.join(words)
def preprocessing(str):
    text = cleaning(normalAt(str))
    text = getJtext(text)
    return text
	
def preprocessing_implentasi(komentar):
    n_komentar = []
    for teks in komentar:
        proses = preprocessing(teks)
        proses = removePunc(replace_word(proses))
        #proses = removePunc(proses)
        proses = stemmer_stopWord(str(proses))
        n_komentar.append(proses)
    return n_komentar



