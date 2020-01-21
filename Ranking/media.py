import Normalizer.Normalizer_Trash_removal as normalizer
import Ranking.rank as rank
import Source_Analyze.analize as aSource
import Sentiment_Analysis.SentimentAnalysis as sentimentAnalysis
from sklearn import svm
import numpy as np
import scipy.stats
import  json
from tkinter import filedialog


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return m, m-h, m+h

inputPath = filedialog.askopenfilename()
jsonStr = open(inputPath).read()
data_json = json.loads(jsonStr)

sizes = []
for registry in data_json:
    registry['title'] = normalizer.normalization(registry['title'])
    registry['title'] = normalizer.tokenize_info(registry['title'])
    registry['title'] = normalizer.stripNonAlphaNum(registry['title'])

    registry['text'] = normalizer.normalization(registry['text'])
    registry['text'] = normalizer.tokenize_info(registry['text'])
    registry['text'] = normalizer.stripNonAlphaNum(registry['text'])

    alltext = registry['text'] + registry['title'] + registry['title']
    sizes.append(len(alltext.split(" ")))

print (mean_confidence_interval(sizes))