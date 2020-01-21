from datetime import datetime
from sklearn import svm
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

inputPath = 'dataset_real_vector_2020.json'
jsonStr = open(inputPath).read()
data_real = json.loads(jsonStr)

inputPathTest = 'dataset_fake_vector_2020.json'
jsonStrTest = open(inputPathTest).read()
data_fake = json.loads(jsonStrTest)

p_words_start = 10
p_words_limit = 1000

include_sentiment_analysis = True
include_source_check = True

colors = {}
colors[100] = 'r'
colors[200] = 'g'
colors[300] = 'b'
max_f1 = 0.0

plot_data_x = []
plot_data_y = []

p_words = p_words_start
while p_words <= p_words_limit:

    # initialize metrics arrays
    accuracy_score = []
    precision_score = []
    recall_score = []
    f1_score = []

    # experiment repetitions
    for seed in range(1):

        training_set = []
        test_set = []
        y_true = []
        y_pred = []
        data_real = shuffle(data_real, random_state=seed)

        # train with real news, test with both fake and real news
        for i in range (len(data_real)):
            if i < len(data_fake):
            #if i < (len(data_real) - len(data_fake)):
                training_set.append(np.asarray(data_real[i]))
            else:
                test_set.append(np.asarray(data_real[i]))
                y_true.append(1) # inlier (real)
        for i in range (len(data_fake)):
            test_set.append(np.asarray(data_fake[i]))
            y_true.append(-1) # outlier (fake)

        # choose appropriate features for the training set
        training_set_final = []
        for x in training_set:
            training_temp = x[:p_words].tolist()
            if (include_sentiment_analysis):
                training_temp.append(x[-1]) # sentiStrengthTitleVector | 0
                training_temp.append(x[-2]) # sentiStrengthTitleVector | 0
                training_temp.append(x[-3]) # sentiStrengthTextVector | 0
                training_temp.append(x[-4]) # sentiStrengthTextVector | 0
                training_temp.append(x[-5]) # textSubjectitivy
                training_temp.append(x[-6]) # textPolarity
                training_temp.append(x[-7]) # titleSubjectivity
                training_temp.append(x[-8]) # titlePolarity
                training_temp.append(x[-9]) # sentiWordsTitle
                training_temp.append(x[-10]) # sentiWordsText
                training_temp.append(x[-11]) # sentimentTitle
                training_temp.append(x[-12]) # sentimentText
            if (include_source_check):
                training_temp.append(x[-13]) # sourceCheck
            training_set_final.append(np.asarray(training_temp))

        # train model
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
        clf.fit(training_set_final)

        # choose appropriate features for the test samples
        for y in test_set:
            test_temp = y[:p_words].tolist()
            test_set_final = []
            if (include_sentiment_analysis):
                test_temp.append(y[-1]) # sentiStrengthTitleVector | 0
                test_temp.append(y[-2]) # sentiStrengthTitleVector | 0
                test_temp.append(y[-3]) # sentiStrengthTextVector | 0
                test_temp.append(y[-4]) # sentiStrengthTextVector | 0
                test_temp.append(y[-5]) # textSubjectitivy
                test_temp.append(y[-6]) # textPolarity
                test_temp.append(y[-7]) # titleSubjectivity
                test_temp.append(y[-8]) # titlePolarity
                test_temp.append(y[-9]) # sentiWordsTitle
                test_temp.append(y[-10]) # sentiWordsText
                test_temp.append(y[-11]) # sentimentTitle
                test_temp.append(y[-12]) # sentimentText
            if (include_source_check):
                test_temp.append(y[-13]) # sourceCheck
            test_set_final.append(np.asarray(test_temp))
            
            # predict output for test sample
            result = clf.predict(test_set_final)
            
            # persist prediction to array
            if result[0] < 0:
                y_pred.append(-1) # is an outlier
            else:
                y_pred.append(1) # is an inlier

        # append metrics to array
        accuracy_score.append(metrics.accuracy_score(y_true, y_pred))
        precision_score.append(metrics.precision_score(y_true, y_pred))
        recall_score.append(metrics.recall_score(y_true, y_pred))
        f1_score.append(metrics.f1_score(y_true, y_pred))
    
    # print datetime and metrics average
    dt = datetime.now()
    dt_str = '{}/{}/{} {}:{}:{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    print("{}\tp_words: {}\tnu: {}\tgamma: {}\taccuracy: {:.4f}\tprecision: {:.4f}\trecall: {:.4f}\tf1: {:.4f}".format(dt_str, p_words, clf.nu, clf.gamma, np.mean(accuracy_score), np.mean(precision_score), np.mean(recall_score), np.mean(f1_score)))

    if np.max(f1_score) > max_f1:
        max_f1 = np.max(f1_score)

    plot_data_x.append(p_words)
    plot_data_y.append(np.mean(f1_score))    

    # increment number of p_words
    p_words = p_words + 10

# plot
print('max f1 score = {}'.format(max_f1))
plt.plot(plot_data_x, plot_data_y)
plt.gca().grid()
plt.show()
