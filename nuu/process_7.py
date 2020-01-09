from datetime import datetime
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

maxWords = 1001

nuu = 0.1
while nuu > 0.0001:

    words = 10
    while words < maxWords:

        accuracy_score = []
        precision_score = []
        recall_score = []
        f1_score = []

        #print(len(data_real))
        #print(len(data_fake))

        for seed in range(1):

            training_set = []
            test_set = []
            y_true = []
            y_pred = []
            data_real = shuffle(data_real, random_state=seed)

            for i in  range (len(data_real)):
                #if i<len(data_real)*0.80:
                if i<(len(data_real)-len(data_fake)):
                    training_set.append(np.asarray(data_real[i]))
                else:
                    test_set.append(np.asarray(data_real[i]))
                    y_true.append(1)

            for i in  range (len(data_fake)):
                test_set.append(np.asarray(data_fake[i]))
                y_true.append(-1)
            
            #print(len(training_set))
            #print(len(test_set))

            training_set_final = []
            for x in training_set:
                #AQUI É FEITO O CONTROLE DO NUMERO DE PALAVRAS QUE ESTAO SENDO USADAS.
                #ARMAZENAR OS "LASTS" DA ANALISE DE FONTE E DE SENTIMENTO
                #EX: 3 ATRIBUTOS DE SENTIMENtO: 5 LAST1,LAST2,LAST3
                #AI TIRA O NUMERO DE PALAVRAS QUE VAI USAR E JUNTA OS LASTS.
                last13 = x[-13] # source check

                toAppendTrain = x[:words].tolist()
                toAppendTrain.append(last13)

                training_set_final.append(np.asarray(toAppendTrain))

            clf = svm.OneClassSVM(nu=nuu, kernel="rbf", gamma='auto')
            clf.fit(training_set_final)

            y_pred = []

            for y in test_set:
                test = []
                last13 = y[-13] # source check

                toAppendTest = y[:words].tolist()
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
        
        dt = datetime.now()
        dt_str = '{}/{}/{} {}:{}:{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        print("{}\twords: {}\tnu: {}\tgamma: {}\taccuracy: {}\tprecision: {}\trecall: {}\tf1: {}".format(dt_str, words, clf.nu, clf.gamma, np.mean(accuracy_score), np.mean(precision_score), np.mean(recall_score), np.mean(f1_score)))

        words = words + 10

    nuu = nuu / 10
