from sklearn import svm
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import json

inputPath = '/home/marcelovca90/git/folder/dataset_real_vector.json'
jsonStr = open(inputPath).read()
data_real = json.loads(jsonStr)

inputPathTest = '/home/marcelovca90/git/folder/dataset_fake_vector.json'
jsonStrTest = open(inputPathTest).read()
data_fake = json.loads(jsonStrTest)

words = 1
maxWords = 1001
while words < maxWords:

    accuracy_score = []
    precision_score = []
    recall_score = []
    f1_score = []

    for seed in range(10):

        training_set = []
        test_set = []
        y_true = []
        y_pred = []
        data_real = shuffle(data_real, random_state=seed)

        for i in  range (len(data_real)):
            if i<len(data_real)*0.50:
                training_set.append(np.asarray(data_real[i]))
            else:
                test_set.append(np.asarray(data_real[i]))
                y_true.append(1)

        for i in  range (len(data_fake)):
            test_set.append(np.asarray(data_fake[i]))
            y_true.append(-1)

        training_set_final = []
        for x in training_set:
            #AQUI É FEITO O CONTROLE DO NUMERO DE PALAVRAS QUE ESTAO SENDO USADAS.
            #ARMAZENAR OS "LASTS" DA ANALISE DE FONTE E DE SENTIMENTO
            #EX: 3 ATRIBUTOS DE SENTIMENtO: 5 LAST1,LAST2,LAST3
            #AI TIRA O NUMERO DE PALAVRAS QUE VAI USAR E JUNTA OS LASTS.
            last1 = x[-1]
            last2 = x[-2]
            last3 = x[-3]
            last4 = x[-4]
            last5 = x[-5]
            last6 = x[-6]
            last7 = x[-7]
            last8 = x[-8]
            last9 = x[-9]
            last10 = x[-10]
            last11 = x[-11]
            last12 = x[-12]
            last13 = x[-13] # source check

            toAppendTrain = x[:words].tolist()
            toAppendTrain.append(last1)
            toAppendTrain.append(last2)
            toAppendTrain.append(last3)
            toAppendTrain.append(last4)
            toAppendTrain.append(last5)
            toAppendTrain.append(last6)
            toAppendTrain.append(last7)
            toAppendTrain.append(last8)
            toAppendTrain.append(last9)
            toAppendTrain.append(last10)
            toAppendTrain.append(last11)
            toAppendTrain.append(last12)
            toAppendTrain.append(last13)

            training_set_final.append(np.asarray(toAppendTrain))

        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
        clf.fit(training_set_final)

        y_pred = []

        for y in test_set:
            test = []
            last1 = y[-1]
            last2 = y[-2]
            last3 = y[-3]
            last4 = y[-4]
            last5 = y[-5]
            last6 = y[-6]
            last7 = y[-7]
            last8 = y[-8]
            last9 = y[-9]
            last10 = y[-10]
            last11 = y[-11]
            last12 = y[-12]
            last13 = y[-13] # source check

            toAppendTest = y[:words].tolist()
            toAppendTest.append(last1)
            toAppendTest.append(last2)
            toAppendTest.append(last3)
            toAppendTest.append(last4)
            toAppendTest.append(last5)
            toAppendTest.append(last6)
            toAppendTest.append(last7)
            toAppendTest.append(last8)
            toAppendTest.append(last9)
            toAppendTest.append(last10)
            toAppendTest.append(last11)
            toAppendTest.append(last12)
            toAppendTest.append(last13)
            
            test.append(np.asarray(toAppendTest))
            
            result = clf.predict(test)
            
            if result[0] < 0:
                y_pred.append(-1)
                # print ('isso é caô.')
            else:
                y_pred.append(1)

        accuracy_score.append(metrics.accuracy_score(y_true, y_pred))
        precision_score.append(metrics.precision_score(y_true, y_pred))
        recall_score.append(metrics.recall_score(y_true, y_pred))
        f1_score.append(metrics.f1_score(y_true, y_pred))
    
    print("words: ", words,", gamma: ", clf.gamma, ", accuracy_score: ", np.mean(accuracy_score), ", precision_score:", np.mean(precision_score), ", recall_score:", np.mean(recall_score), ", f1_score:", np.mean(f1_score))

    words = words + 1