import numpy as np
import pandas as pd
import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def analyze(name, MinLength, MaxLength, MaxIteration):
    global logkeeper
    #  C:\Users\axdzh\AppData\Local\Programs\Python\Python38
    dataPath = os.path.abspath("C:/Users/axdzh/Desktop/kevin/projects/data/"
                               + name
                               + ".csv")
    print(dataPath)
    stockDataRaw = pd.read_csv(dataPath)
    stockDataRaw = pd.DataFrame(stockDataRaw[:-1])
    # print(stockDataRaw.head())
    stockData = stockDataRaw.filter([
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Attitude For Next Day', 'Attitude For Next Week'])
    # print(stockData.iloc[0:7, :5])
    # print(stockData.iloc[6:, -2])
    for lengthIteration in range(MinLength, MaxLength+1):
        predictionLength = lengthIteration
        # First line, becasue appending to an
        # empty dataframe "doesn't just work".
        x = np.array([stockData.iloc[0:predictionLength, :5]
                     .to_numpy()
                     .flatten()])
        y = stockData.iloc[predictionLength-1:, -2]
        for i in range(1, len(stockData)-predictionLength+1):
            tempo = stockData.iloc[i:i+predictionLength, :5].to_numpy()
            tempo = tempo.flatten()
            x = np.vstack((x, [tempo]))
        x = pd.DataFrame(x)
        print(x)
        # print(x[0:100])
        # print(y)
        # for metaIteration in range(1, MaxIteration+1):
        metaIteration = MaxIteration
        total_accuracy = 0
        x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.1)
        # classifier = make_pipeline(
        #         StandardScaler(), BaggingClassifier(
        #             DecisionTreeClassifier(
        #                 splitter='random'),
        #             n_estimators=800, n_jobs=-4))
        classifier = make_pipeline(StandardScaler(), SGDClassifier(
                 loss='perceptron', max_iter=1000, tol=1e-3, n_jobs=-3))
        for j in range(metaIteration):
            classifier.fit(x_train, y_train)
            # Literally from stackoverflow
            # nsamples, nx, ny = x_train.shape
            # x_train = np.reshape((nsamples, nx*ny))
            #    classifier = DecisionTreeClassifier(random_state=42)
            #    classifier = BaggingClassifier(DecisionTreeClassifier(
            # splitter='random', random_state=42), n_estimators=500, n_jobs=-3)
            #  classifier = make_pipeline(StandardScaler(), SGDClassifier(
            #     loss='perceptron', max_iter=1000, tol=1e-3, n_jobs=-3))
            y_prediction = classifier.predict(x_test)
            print(confusion_matrix(y_test, y_prediction))
            temp_accuracy = accuracy_score(y_test, y_prediction)
            print(
                temp_accuracy, ' ----------------Prediction Length: ',
                predictionLength, ' ----------------MetaIteration: ',
                metaIteration, ' ----------------Real Iteration: ', j+1)
            total_accuracy += temp_accuracy
        avgAcc = total_accuracy/metaIteration
        logkeeper = logkeeper.append(pd.Series([predictionLength,
                                                metaIteration, avgAcc],
                                               index=logkeeper.columns),
                                     ignore_index=True)
        logkeeper.to_csv("C:/Users/axdzh/Desktop/kevin/projects/data/"
                         + 'MSFT'
                         + "log.csv")

    print(avgAcc)
    return

# Going one level higher for Debugging purposes $ stuff,
# turned out to be a never used feature, really.


def metaAnalyze(name, MinLength, MaxLength, MaxIteration):
    global logkeeper
    dataPath = os.path.abspath("C:/Users/axdzh/Desktop/kevin/projects/data/"
                               + name
                               + ".csv")
    print(dataPath)
    stockDataRaw = pd.read_csv(dataPath)
    stockDataRaw = pd.DataFrame(stockDataRaw[:-1])
    stockData = stockDataRaw.filter([
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Attitude For Next Day', 'Attitude For Next Week'])
    for lengthIteration in range(MinLength, MaxLength+1):
        predictionLength = lengthIteration
        x = np.array(stockData.iloc[0:predictionLength-1, 5])
        y = stockData.iloc[predictionLength-1:, -2]
        for i in range(1, len(stockData)-predictionLength+1):
            tempo = stockData.iloc[i:i+predictionLength-1, 5].to_numpy()
            x = np.vstack((x, tempo))
        x = pd.DataFrame(x)
        # print(x[0:100])
        x = x.replace('YES', 1)
        x = x.replace('NO', 0)
        print(x[0:100])
        metaIteration = MaxIteration
        total_accuracy = 0
        x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.1)
        classifier = classifier = make_pipeline(SGDClassifier(
                 loss='log', max_iter=1000, tol=1e-3, n_jobs=-3))
        for j in range(metaIteration):
            classifier.fit(x_train, y_train)
            y_prediction = classifier.predict(x_test)
            print(confusion_matrix(y_test, y_prediction))
            temp_accuracy = accuracy_score(y_test, y_prediction)
            print(temp_accuracy, ' ----------------Prediction Length: ',
                  predictionLength, ' ----------------MetaIteration: ',
                  metaIteration, ' ----------------Real Iteration: ', j+1)
            total_accuracy += temp_accuracy
        avgAcc = total_accuracy/metaIteration
        logkeeper = logkeeper.append(pd.Series([predictionLength,
                                                metaIteration, avgAcc],
                                               index=logkeeper.columns),
                                     ignore_index=True)
        logkeeper.to_csv("C:/Users/axdzh/Desktop/kevin/projects/data/" + 'MSFT' + "metalog.csv")
    print(avgAcc)
    return


logkeeper = pd.DataFrame(columns=['Prediction Length', 'Iterations',
                                  'Average Accuracy per Iteration'])
# analyze('TSLA', 1, 7, 20)
metaAnalyze('MSFT', 20, 20, 1)
# analyze('AAPL', 1, 7, 20)
