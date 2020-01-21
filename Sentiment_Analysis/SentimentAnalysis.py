# -*- coding: utf-8 -*-
"""
Análise de sentimento
TRABALHO DE CONCLUSÃO DE CURSO - 2 sem. 2018.
"Análise da influência de Atributos de Sentimento na Identificação de Fake News"
"""
import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import subprocess
import shlex

#análise do sentistrength via cmd.
def RateSentiment(sentiString):
    p = subprocess.Popen(shlex.split("java -jar C:/SentiStrength.jar stdin sentidata C:/SentStrength_Data/"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    b = bytes(sentiString.replace(" ","+"), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")
    stdout_text = stdout_text.rstrip().replace("\t"," ")
    return stdout_text #+ " " + sentiString


def StanfordSentiment(sentiString):
    #open a subprocess using shlex to get the command line string into the correct args list format
    #Modify the location of SentiStrength.jar and D:/SentiStrength_Data/ if necessary
    p = subprocess.Popen(shlex.split("java -cp '*' -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -stdin"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #communicate via stdin the string to be rated. Note that all spaces are replaced with +
    #Can't send string in Python 3, must send bytes
    c = bytes(sentiString, 'utf-8')
    stdout_byte, stderr_text = p.communicate(c)
    #convert from byte
    stdout_text = stdout_byte.decode("utf-8")
    #replace the tab with a space between the positive and negative ratings. e.g. 1    -5 -> 1 -5
    stdout_text = stdout_text.rstrip().replace("\r\n"," ")
    if stdout_text == "Positive":
        return 1
    elif stdout_text == "Negative":
        return -1
    return 0

# Analisa o sentimento do parágrafo inteiro e retorna um valor numérico representando o grau sentimento nltk
def sentiment_value_per_paragraph(paragraph):
    analyser = SentimentIntensityAnalyzer()
    result = analyser.polarity_scores(paragraph)
    score = result['compound']
    return round(score,1)

    """
    [ -1, -0.5) : 1, V.Negative
    [-0.5, 0) : 2, Negative
    [0] : 3, Neutral
    (0, 0.5) : 4, Positive
    [0.5, 1] : 5, V.Positive
    [] inclusive
    () exclusive
    """

# Analisa o sentimento de cada frase do parágrafo e retorna as médias dos valores de grau de sentimento de cada frase nltk
def sentiment_value_per_phrase(paragraph):
    analyser = SentimentIntensityAnalyzer()
    text = nltk.sent_tokenize(paragraph)
    score = 0
    for phrase in text:
        result = analyser.polarity_scores(phrase)
        score = score + result['compound']
        print (phrase + " = " + str(result['compound']))
    score /= len(text)
    return round(score,1)

    """
    [ -1, -0.5) : 1, V.Negative
    [-0.5, 0) : 2, Negative
    [0] : 3, Neutral
    (0, 0.5) : 4, Positive
    [0.5, 1] : 5, V.Positive
    [] inclusive
    () exclusive
    """

# Analisa o sentimento do parágrafo inteiro e retorna um valor numérico representando o sentimento (-1, 0 ou 1)
def sentiment_per_paragraph(paragraph):
    analyser = SentimentIntensityAnalyzer()
    result = analyser.polarity_scores(paragraph)
    score = result['compound']
    score = round(score,1)
    if score < -0.1:
        return -1
    if score > 0.1:
        return 1
    else:
        return 0

    """
    -1:  Negative
    0 :  Neutral
    1 :  Positive
    """

# Analisa o sentimento de cada frase do parágrafo e retorna o sentimento médio do parágrafo (-1, 0 ou 1)
def sentiment_per_phrase(paragraph):
    analyser = SentimentIntensityAnalyzer()
    text = nltk.sent_tokenize(paragraph)
    score = 0
    for phrase in text:
        result = analyser.polarity_scores(phrase)
        score = score + result['compound']
        print (phrase + " = " + ( "Positivo" if (result['compound'] > 0) else ("Negativo" if (result['compound'] < 0) else "Neutro")))
    score /= len(text)
    score = round(score,1)
    if score < -0.1:
        return -1
    if score > 0.1:
        return 1
    else:
        return 0

    """
    -1:  Negative
    0 :  Neutral
    1 :  Positive
    """

def main():
    # Parágrafo usado como exemplo 
    sample = "I loved this one. But I didn't like that one."

    print("\nSample = " + sample + '\n')
    sent = sentiment_value_per_paragraph(sample)
    print('Sentiment value of whole paragraph: ' + str(sent))
    sent = sentiment_per_paragraph(sample)
    print('Sentiment whole paragraph: ' + ( "Positivo" if (sent == 1) else ("Negativo" if (sent == -1) else "Neutro")) + '\n')
    sent = sentiment_value_per_phrase(sample)
    print('Sentiment average value of each phrase: ' + str(sent) + '\n')
    sent = sentiment_per_phrase(sample)
    print('Sentiment average of each phrase: ' + ( "Positivo" if (sent == 1) else ("Negativo" if (sent == -1) else "Neutro")))

if __name__ == "__main__":
    main()