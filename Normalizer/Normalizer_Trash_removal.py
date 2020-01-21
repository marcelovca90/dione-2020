import json
import re
import unicodedata
import tkinter as tk
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#nltk.download('stopwords')
file_opt={}

def unicode_equivalence(data):
    nfkd = unicodedata.normalize('NFKD', data)
    data = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    return data

def to_lower_case(data):
    return data.lower()

def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)
    filtered_data = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_data)

def strip_non_alphanumeric(data):
    sentence = data
    sentence = re.sub('[^a-zA-Z0-9 \\\]', '', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    return sentence.strip()

def replace_numbers(data):
    words = []
    for word in data.split():
        if word.isdigit():
            words.append('somenumber')
        else:
            words.append(word)
    return ' '.join(words)

#Tokenização do campo "informação", removendo stopwords
def tokenize_info(info):
    
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(info)
    words_filtered = []
    
    for w in words:
        if w == 'a' or w == 'an' or w == 'the':
            words_filtered.append('articletoken')
        elif w.isdigit():
            words_filtered.append('numbertoken')
        elif w not in stopWords:
            words_filtered.append(w)
    
    info_tokenized = ""
    
    for w in words_filtered:
        info_tokenized = info_tokenized + " " + w

    return info_tokenized

#Criação do arquivo Json filtrado e normalizado
def create_normalized_Json(title,author,info_tokenized,url):
    
    name_arq = filedialog.asksaveasfilename(**file_opt)
    arq = open(name_arq,'w')
    data = {'title':title,'author':author,'information':info_tokenized,'url':url}
    json.dump(data,arq) 
    arq.close()
    return 

#Faz a leitura do Json no padrão do Dataset "https://github.com/KaiDMML/FakeNewsNet"
def read_Json():
    
    root = tk.Tk()
    root.withdraw()

    inputPath = filedialog.askopenfilename()
    data_json = open(inputPath).read()
    data_jsonInfo = json.loads(data_json)
    
    title = data_jsonInfo.get('meta_data').get('og').get('title')
    author = data_jsonInfo.get('authors')
    author = str(author).replace('[','')
    author = str(author).replace(']','')
    info = data_jsonInfo.get('meta_data').get('og').get('description')
    url = data_jsonInfo.get('meta_data').get('og').get('url')
    
    info = normalization(info)
    info = stripNonAlphaNum(info)    

    info_tokenized = tokenize_info(info)

    create_normalized_Json(title,author,info_tokenized,url)
    
    print('Title: ',title)
    print('Author name: ',author)
    print('Information: ',info_tokenized)
    print('Url: ',url)
