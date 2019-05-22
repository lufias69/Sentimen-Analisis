import re
alamata = ['ID-OpinionWords-master/positive.txt', 'ID-OpinionWords-master/negative.txt']
def getData(alamat):
    lineList = list()
    #with open(dir_path + '/' + alamat) as f:
    with open(alamat) as f:
        for line in f:
            lineList.append(line.rstrip('\n'))
    return lineList


kamus_ = list()
for i in alamata:
    kamus_+=getData(i)
    
kamus = list()
for i in kamus_:
    i = i.replace("tdk",'tidak').replace("yng",'yang').replace("yg",'yang').replace("dg",'dengan').replace("dgn",'dengan')
    i = i.replace(" ", "_")
    kamus.append(i)
    i = i.split()
    if i[-1]=='-':
        print(i)
        
def index_ngram (kata, n):
    len_kata = len(kata.split())
    #print(len_kata)
    idx = list()
    return_index = list()
    n = n
    for i in range(0,len_kata,1):
        idx.append(i)

    for i, ix in enumerate(idx):
        if n ==2:
            if i < len_kata-1:
                l = [idx[i],idx[i+1]]
                return_index.append(l)
        elif n ==3 :
            if i < len_kata-2:
                l = [idx[i],idx[i+1], idx[i+2]]
                return_index.append(l)
    return return_index
#index_ngram ("ajar baik bicara terlalu cepat jelas mudah paham")

def bersihkan(komentar):
    i = komentar
    i = i.replace("!","").replace("?","")
    i = i.replace(">","").replace("<","").replace("%","").replace("%","").replace("@","").replace("*","").replace(";","").replace(":","").replace("'","").replace(")","").replace("(","")
    i = i.replace("^","").replace(".","")
    return i

def ngram (teks, n=2):
    index = index_ngram (teks, n)
    if len(teks.split()) > 2:
        index += index_ngram (teks, 3)
    #print(index)
    split_teks = teks.split()
    for i in index:
        kata_cek  =  "_".join(split_teks[i[0]:i[-1]+1])
        #print(kata_cek)
        if kata_cek in kamus:
            for j in i:
                split_teks[j]=""
            split_teks[i[0]] = kata_cek
    nretun = " ".join(split_teks).lstrip().rstrip()
    xxx = re.sub(' +', ' ',nretun)
    return bersihkan(xxx)

#ngram("berbaik hati belas kasihan bersemangat_meluap-luap", 2)


