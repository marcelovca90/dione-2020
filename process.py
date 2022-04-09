from datetime import datetime
from sklearn import svm
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

inputPath = 'dataset_real_vector_2020_normalized.json'
jsonStr = open(inputPath).read()
data_real = json.loads(jsonStr)

inputPathTest = 'dataset_fake_vector_2020_normalized.json'
jsonStrTest = open(inputPathTest).read()
data_fake = json.loads(jsonStrTest)

p_words_start = 1
p_words_limit = 1000

include_sentiment_analysis = False
include_source_check = False

plot_data_x = []
plot_data_y_precision = []
plot_data_y_recall = []
plot_data_y_f1 = []

p_words = p_words_start
while p_words <= p_words_limit:

    # initialize metrics arrays
    accuracy_score = []
    precision_score = []
    recall_score = []
    f1_score = []

    # experiment repetitions
    for seed in range(1):

        X_train_temp = []
        X_test_temp = []
        Y_train_temp = []
        Y_test_temp = []

        X_train = []
        X_test = []
        Y_train = []
        Y_test = []

        data_real = shuffle(data_real, random_state=seed)

        # train with real news, test with both fake and real news
        for i in range (len(data_real)):
            if i < (len(data_real) - len(data_fake)):
                X_train_temp.append(np.asarray(data_real[i]))
                Y_train_temp.append(1) # inlier (real)
            else:
                X_test_temp.append(np.asarray(data_real[i]))
                Y_test_temp.append(1) # inlier (real)
        for i in range (len(data_fake)):
            X_test_temp.append(np.asarray(data_fake[i]))
            Y_test_temp.append(-1) # outlier (fake)

        # choose appropriate features for the training set
        for x in X_train_temp:
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
            X_train.append(training_temp)
        Y_train = Y_train_temp

        # train model
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
        clf.fit(X_train)

        # choose appropriate features for the test samples
        for x in X_test_temp:
            test_temp = x[:p_words].tolist()
            if (include_sentiment_analysis):
                test_temp.append(x[-1]) # sentiStrengthTitleVector | 0
                test_temp.append(x[-2]) # sentiStrengthTitleVector | 0
                test_temp.append(x[-3]) # sentiStrengthTextVector | 0
                test_temp.append(x[-4]) # sentiStrengthTextVector | 0
                test_temp.append(x[-5]) # textSubjectitivy
                test_temp.append(x[-6]) # textPolarity
                test_temp.append(x[-7]) # titleSubjectivity
                test_temp.append(x[-8]) # titlePolarity
                test_temp.append(x[-9]) # sentiWordsTitle
                test_temp.append(x[-10]) # sentiWordsText
                test_temp.append(x[-11]) # sentimentTitle
                test_temp.append(x[-12]) # sentimentText
            if (include_source_check):
                test_temp.append(x[-13]) # sourceCheck
            X_test.append(test_temp)
        Y_test = Y_test_temp

        #print('{} {} {} {}'.format(len(X_train), len(X_test), len(Y_train), len(Y_test)))
            
        # predict and persist outputs for test samples
        # -1 is an outlier, +1 is an inlier
        Y_pred = clf.predict(X_test)

        # append metrics to array
        accuracy_score.append(metrics.accuracy_score(Y_test, Y_pred))
        precision_score.append(metrics.precision_score(Y_test, Y_pred))
        recall_score.append(metrics.recall_score(Y_test, Y_pred))
        f1_score.append(metrics.f1_score(Y_test, Y_pred))
    
    # print datetime and metrics average
    dt = datetime.now()
    dt_str = '{}/{}/{} {}:{}:{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    print("{}\tp_words: {}\tnu: {}\tgamma: {}\taccuracy: {:.4f}\tprecision: {:.4f}\trecall: {:.4f}\tf1: {:.4f}".format(dt_str, p_words, clf.nu, clf.gamma, np.mean(accuracy_score), np.mean(precision_score), np.mean(recall_score), np.mean(f1_score)))

    plot_data_x.append(p_words)
    plot_data_y_precision.append(np.mean(precision_score))
    plot_data_y_recall.append(np.mean(recall_score))
    plot_data_y_f1.append(np.mean(f1_score))

    # increment number of p_words
    p_words = p_words + 10

# plot
fig,ax = plt.subplots()

plot_data_y_precision_mean = [np.mean(plot_data_y_precision) for i in plot_data_x]
ax.plot(plot_data_x, plot_data_y_precision, label='precision', color='darkred')
ax.plot(plot_data_x, plot_data_y_precision_mean, label='precision (mean)', color='red', linestyle='--')
print('precision (mean) = ', plot_data_y_precision_mean[0])

plot_data_y_recall_mean = [np.mean(plot_data_y_recall) for i in plot_data_x]
ax.plot(plot_data_x, plot_data_y_recall, label='recall', color='darkgreen')
ax.plot(plot_data_x, plot_data_y_recall_mean, label='recall (mean)', color='green', linestyle='--')
print('recall (mean) = ', plot_data_y_recall_mean[0])

plot_data_y_f1_mean = [np.mean(plot_data_y_f1) for i in plot_data_x]
ax.plot(plot_data_x, plot_data_y_f1, label='f1', color='darkblue')
ax.plot(plot_data_x, plot_data_y_f1_mean, label='f1 (mean)', color='blue', linestyle='--')
print('f1 (mean) = ', plot_data_y_f1_mean[0])

legend = ax.legend(loc='best')
plt.gca().grid()
plt.show()
