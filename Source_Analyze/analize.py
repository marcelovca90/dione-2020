import json
import requests
from tkinter import filedialog
from urllib.parse import urlparse
import os

def analizeSource(data):

    # copia a URL completa para a var url para ser tratada
    if data:
        url = data
    else :
        return 0 

    # divide a URL em partes ex.:(scheme='http', netloc='www.cwi.nl:80', path='/%7Eguido/Python.html', params='', query='', fragment='')
    o = urlparse(url)

    # concatena apenas as partes interessantes da URL completa
    url = '{uri.scheme}://{uri.netloc}/'.format(uri=o)

    target = '{uri.netloc}'.format(uri=o)

    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = 'allSources.txt'
    abs_file_path = os.path.join(script_dir, rel_path)

    arq = open(abs_file_path,'r+').read()

    data_json = json.loads(arq)

    for registry in data_json:
        if registry['url'] and registry['url'] == url :
            return registry['trustworthiness']
            
    # Corpo da requisição GET
    payload = {'hosts': url, 'callback': '', 'key' : '4175cd3b4e4e52fb19c4037aa6f776ba60b175c6'}

    # requisição GET
    res = requests.get(' http://api.mywot.com/0.4/public_link_json2', params = payload)

    #retorna a resposta da requisição
    if ((json.loads((res.text.split("(")[1]).split(")")[0])).get(target)).get('0') :
        value = ((json.loads((res.text.split("(")[1]).split(")")[0])).get(target)).get('0')[0]

    else: 
        value = 0

    if value < 60 :
        trustworthiness = 0
    if value >= 60 :
        trustworthiness = 1

    dataFile = {"url" : url, "trustworthiness" : trustworthiness}
    json.dump(dataFile,arq) 
    arq.close()

    return trustworthiness
