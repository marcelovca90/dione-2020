import Normalizer.Normalizer_Trash_removal as normalizer
import Ranking.rank as rank
import Source_Analyze.analize as aSource
import Sentiment_Analysis.SentimentAnalysis as sentimentAnalysis
from textblob import TextBlob
from sklearn import svm
from sklearn.preprocessing import normalize
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import json

words_to_rank = [line.rstrip('\n') for line in open('words.txt')]

inputPath = 'database/Real.json'
jsonStr = open(inputPath).read()
data_json = json.loads(jsonStr)
data_set = []

generate_word_ranking = False
word_dict = {}
word_count = 0
news_count = 0
total_news = len(data_json)

for registry in data_json:
    wordCount = []
    sentimentText = sentimentAnalysis.sentiment_value_per_paragraph(registry['text'])
    #print('\nsentimentText = {}'.format(sentimentText))
    sentimentTitle = sentimentAnalysis.sentiment_value_per_paragraph(registry['title'])
    #print('\nsentimentTitle = {}'.format(sentimentTitle))

    registry['title'] = normalizer.unicode_equivalence(registry['title'])
    registry['title'] = normalizer.to_lower_case(registry['title'])
    registry['title'] = normalizer.remove_stop_words(registry['title'])
    registry['title'] = normalizer.strip_non_alphanumeric(registry['title'])
    registry['title'] = normalizer.replace_numbers(registry['title'])
    #print('\nregistry[\'title\'] = [{}]'.format(registry['title']))

    registry['text'] = normalizer.unicode_equivalence(registry['text'])
    registry['text'] = normalizer.to_lower_case(registry['text'])
    registry['text'] = normalizer.remove_stop_words(registry['text'])
    registry['text'] = normalizer.strip_non_alphanumeric(registry['text'])
    registry['text'] = normalizer.replace_numbers(registry['text'])
    #print('\nregistry[\'text\'] = [{}]'.format(registry['text']))

    wordCount = rank.get_key_words_count(registry['text'] + registry['title'] + registry['title'], words_to_rank[:1000])

    wordCount.append(aSource.analizeSource(registry['url']))
    #print(sentimentTitle)
    #sentimento do nltk 2
    wordCount.append(sentimentText)
    wordCount.append(sentimentTitle)

    #contagem de emotionWords 2
    wordCount.append(rank.get_senti_words_count(registry['text']))
    wordCount.append(rank.get_senti_words_count(registry['title']))

    #subjetividade e polaridade 4
    testimonialTitle = TextBlob(registry['title'])
    #print(testimonialTitle.sentiment)
    wordCount.append(testimonialTitle.sentiment.polarity)
    wordCount.append(testimonialTitle.sentiment.subjectivity)
    testimonialText = TextBlob(registry['text'])
    #print(testimonialText.sentiment)
    wordCount.append(testimonialText.sentiment.polarity)
    wordCount.append(testimonialText.sentiment.subjectivity)

    #sentiStrength 4
    sentiStrengthTextFull = sentimentAnalysis.RateSentiment(registry['text'])
    sentiStrengthTextVector = word_tokenize(sentiStrengthTextFull)
    if len(sentiStrengthTextVector)>0:
        wordCount.append(sentiStrengthTextVector[0])
        wordCount.append(sentiStrengthTextVector[1])
    else:
        wordCount.append(0)
        wordCount.append(0)
    sentiStrengthTitleFull = sentimentAnalysis.RateSentiment(registry['title'])
    sentiStrengthTitleVector = word_tokenize(sentiStrengthTitleFull)
    if len(sentiStrengthTitleVector)>0:
        wordCount.append(sentiStrengthTitleVector[0])
        wordCount.append(sentiStrengthTitleVector[1])
    else:
        wordCount.append(0)
        wordCount.append(0)

    sample = np.asarray(wordCount)
    sample_normalized = normalize(sample[:,np.newaxis], axis=0).ravel() 
    data_set.append(sample_normalized.tolist())

    news_count += 1

    if generate_word_ranking:
        registry_title_words = registry['title'].split()
        word_count += len(registry_title_words)
        for word in registry_title_words:
            if word not in word_dict.keys():
                word_dict[word] = 1
            else:    
                word_dict[word] = word_dict[word] + 1
        registry_text_words = registry['text'].split()
        word_count += len(registry_text_words)
        for word in registry_text_words:
            if word not in word_dict.keys():
                word_dict[word] = 1
            else:    
                word_dict[word] = word_dict[word] + 1
        print ('{}/{} ({} + {} = {} words, average = {:0.3f})'.format(news_count, total_news, len(registry_title_words), len(registry_text_words), len(registry_title_words) + len(registry_text_words), word_count / news_count))

    else:
        print ('{}/{}'.format(news_count, total_news))

f = open('dataset_real_vector_2020.json','w')
json.dump(data_set,f)
f.close()

if generate_word_ranking:

    diff_words = len(word_dict.keys())
    print('number of different words = {}'.format(diff_words))

    avg_words = word_count / news_count
    print('average number of words per news = {}'.format(avg_words))

    word_ranking = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)

    f = open("occurrences.txt", "w")
    for key_value in word_ranking:
        f.write('{}\t{}\n'.format(key_value[0], key_value[1]))
    f.close()

    f = open("words.txt", "w")
    for key_value in word_ranking:
        f.write('{}\n'.format(key_value[0]))
    f.close()