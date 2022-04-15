# -*- coding: utf-8 -*-
"""Bert_Legal_Bert.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_6lpLO1TPysiF0P2rmLV7XkOGajXSkPO

## Data input and Visualisation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')

file_name = "BERT_base_512"
with open("logs"+file_name+".csv", "a") as myfile:
    myfile.write("\nBERT_base_512")
    myfile.write("\nTrain Precision,Train Recall,Train F1,Test Precision,Test Recall,Test F1,Time")
mydata = pd.read_csv('averaging_legalbert_2560.csv')

print(mydata.loc[mydata['label'] == 0].shape)
print(mydata.loc[mydata['label'] == 1].shape)

random_state = 42
# mydata = mydata.sample(frac=1, random_state=random_state)
# mydata = mydata.iloc[:400]
mydata.drop(['uid'], axis=1, inplace=True)

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.naive_bayes import N

# pip install LightGBM
from lightgbm import LGBMClassifier

X_train, X_test, Y_train, Y_test = train_test_split(mydata.drop(['label'], axis=1), mydata['label'], test_size=0.1,
                                                    random_state=random_state, stratify=mydata['label'])

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

"""## Loop for finding Hyper Parameters of SVM"""

with open("logs"+file_name+".csv", "a") as myfile:
    myfile.write("\n"+file_name+"_SVM")

cross_validation_scores = []  # Cross Validation Score
train_confus_matrix = []  # Confusion Matrix
test_confus_matrix = []  # Confusion Matrix

train_preREf1 = []
test_preREf1 = []

train_acc = []
test_acc = []  # Test Score

i = 0
for kernel in ['linear', 'rbf', 'poly']:
    for C in [0.01, 0.1, 1.0]:
        svc = svm.SVC(kernel=kernel, C=C, gamma='scale', random_state=random_state, class_weight='balanced')

        svc.fit(X_train, Y_train)
        # cross_validation = cross_val_score(rfc,X_train,Y_train,scoring="accuracy",cv=5)
        # cross_validation_scores.append(cross_validation.mean())
        i += 1
        print(i)
        predictions = svc.predict(X_train)

        trainReport = classification_report(Y_train.values, predictions, output_dict=True)
        testReport = classification_report(Y_test.values, svc.predict(X_test), output_dict=True)
        train_preREf1.append({'precision': format(trainReport['macro avg']['precision'], '.2f'),
                              'recall': format(trainReport['macro avg']['recall'], '.2f'),
                              'f1Score': format(trainReport['macro avg']['f1-score'], '.2f')})
        test_preREf1.append({'precision': format(testReport['macro avg']['precision'], '.2f'),
                             'recall': format(testReport['macro avg']['recall'], '.2f'),
                             'f1Score': format(testReport['macro avg']['f1-score'], '.2f')})

        train_confus_matrix.append(confusion_matrix(Y_train.values, predictions))
        test_confus_matrix.append(confusion_matrix(Y_test.values, svc.predict(X_test)))

        train_acc.append(svc.score(X_train, Y_train))
        test_acc.append(svc.score(X_test, Y_test))

        with open("logs"+file_name+".csv", "a") as myfile:
            myfile.write("\n" + str(format(trainReport['macro avg']['precision'], '.2f')) + "," + str(
                format(trainReport['macro avg']['recall'], '.2f'))
                         + "," + str(format(trainReport['macro avg']['f1-score'], '.2f')) + "," + str(
                format(testReport['macro avg']['precision'], '.2f'))
                         + "," + str(format(testReport['macro avg']['recall'], '.2f')) + "," + str(
                format(testReport['macro avg']['f1-score'], '.2f')) + "," + time.ctime())

parameters = []

for kernel in ['linear', 'rbf', 'poly']:
    for C in [0.01, 0.1, 1.0]:
        dictonary = {'kernel': kernel, 'C': C}
        parameters.append(dictonary)

trainLab0Score = []
trainLab1Score = []

for matrix in train_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

testLab0Score = []
testLab1Score = []

for matrix in test_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    testLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    testLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

Acc_Table = pd.DataFrame({"Parameters": parameters, "Train Accuracy": train_acc, "Train Label 0 Scores": trainLab0Score,
                          "Train label 1 Scores": trainLab1Score, "Train Avg Scores": train_preREf1,
                          "Test Accuracy": test_acc,
                          "Test Label 0 Scores": testLab0Score, "Test Label 1 Scores": testLab1Score,
                          "Test Avg Scores": test_preREf1}).sort_values(['Test Accuracy'], ascending=False)
Acc_Table.to_csv(file_name + "SVC.csv", index=False)

"""## Loop for finding Hyper Parameters of Random Forest"""

with open("logs"+file_name+".csv", "a") as myfile:
    myfile.write("\n"+file_name+"_RFC")

cross_validation_scores = []  # Cross Validation Score
train_confus_matrix = []  # Confusion Matrix
test_confus_matrix = []  # Confusion Matrix

train_preREf1 = []
test_preREf1 = []

train_acc = []
test_acc = []  # Test Score

i = 0
for n_estimators in [100, 200, 400]:
    for min_samples_split in [15, 30, 50]:
        for max_depth in [2, 3, 5]:
            rfc = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                         n_estimators=n_estimators, class_weight='balanced', random_state=42)
            rfc.fit(X_train, Y_train)
            # cross_validation = cross_val_score(rfc,X_train,Y_train,scoring="accuracy",cv=5)
            # cross_validation_scores.append(cross_validation.mean())
            i += 1
            print(i)
            predictions = rfc.predict(X_train)

            trainReport = classification_report(Y_train.values, predictions, output_dict=True)
            testReport = classification_report(Y_test.values, rfc.predict(X_test), output_dict=True)
            train_preREf1.append({'precision': format(trainReport['macro avg']['precision'], '.2f'),
                                  'recall': format(trainReport['macro avg']['recall'], '.2f'),
                                  'f1Score': format(trainReport['macro avg']['f1-score'], '.2f')})
            test_preREf1.append({'precision': format(testReport['macro avg']['precision'], '.2f'),
                                 'recall': format(testReport['macro avg']['recall'], '.2f'),
                                 'f1Score': format(testReport['macro avg']['f1-score'], '.2f')})

            train_confus_matrix.append(confusion_matrix(Y_train.values, predictions))
            test_confus_matrix.append(confusion_matrix(Y_test.values, rfc.predict(X_test)))

            train_acc.append(rfc.score(X_train, Y_train))
            test_acc.append(rfc.score(X_test, Y_test))

            with open("logs"+file_name+".csv", "a") as myfile:
                myfile.write("\n" + str(format(trainReport['macro avg']['precision'], '.2f')) + "," + str(
                    format(trainReport['macro avg']['recall'], '.2f'))
                             + "," + str(format(trainReport['macro avg']['f1-score'], '.2f')) + "," + str(
                    format(testReport['macro avg']['precision'], '.2f'))
                             + "," + str(format(testReport['macro avg']['recall'], '.2f')) + "," + str(
                    format(testReport['macro avg']['f1-score'], '.2f')) + "," + time.ctime())

parameters = []
for n_estimators in [100, 200, 400]:
    for min_samples_split in [15, 30, 50]:
        for max_depth in [2, 3, 5]:
            dictonary = {'n_estimators': n_estimators, 'min_samples_split': min_samples_split, 'max_depth': max_depth}
            parameters.append(dictonary)

trainLab0Score = []
trainLab1Score = []

for matrix in train_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

testLab0Score = []
testLab1Score = []

for matrix in test_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    testLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    testLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

Acc_Table = pd.DataFrame({"Parameters": parameters, "Train Accuracy": train_acc, "Train Label 0 Scores": trainLab0Score,
                          "Train label 1 Scores": trainLab1Score, "Train Avg Scores": train_preREf1,
                          "Test Accuracy": test_acc,
                          "Test Label 0 Scores": testLab0Score, "Test Label 1 Scores": testLab1Score,
                          "Test Avg Scores": test_preREf1}).sort_values(['Test Accuracy'], ascending=False)
Acc_Table.to_csv(file_name+"RFC.csv", index=False)

"""## Loop for finding Hyper Parameters of Logistic Regression """

with open("logs"+file_name+".csv", "a") as myfile:
    myfile.write("\n"+file_name+"_LR")

cross_validation_scores = []  # Cross Validation Score
train_confus_matrix = []  # Confusion Matrix
test_confus_matrix = []  # Confusion Matrix

train_preREf1 = []
test_preREf1 = []

train_acc = []
test_acc = []  # Test Score

i = 0
for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag']:
    for C in [0.01, 0.05, 0.1, 1.0, 5.0, 10.0]:
        lr = LogisticRegression(solver=solver, C=C, class_weight='balanced',
                                random_state=random_state)  # ,class_weight='balanced')

        lr.fit(X_train, Y_train)
        # cross_validation = cross_val_score(rfc,X_train,Y_train,scoring="accuracy",cv=5)
        # cross_validation_scores.append(cross_validation.mean())
        i += 1
        print(i)
        predictions = lr.predict(X_train)

        trainReport = classification_report(Y_train.values, predictions, output_dict=True)
        testReport = classification_report(Y_test.values, lr.predict(X_test), output_dict=True)
        train_preREf1.append({'precision': format(trainReport['macro avg']['precision'], '.2f'),
                              'recall': format(trainReport['macro avg']['recall'], '.2f'),
                              'f1Score': format(trainReport['macro avg']['f1-score'], '.2f')})
        test_preREf1.append({'precision': format(testReport['macro avg']['precision'], '.2f'),
                             'recall': format(testReport['macro avg']['recall'], '.2f'),
                             'f1Score': format(testReport['macro avg']['f1-score'], '.2f')})

        train_confus_matrix.append(confusion_matrix(Y_train.values, predictions))
        test_confus_matrix.append(confusion_matrix(Y_test.values, lr.predict(X_test)))

        train_acc.append(lr.score(X_train, Y_train))
        test_acc.append(lr.score(X_test, Y_test))

        with open("logs"+file_name+".csv", "a") as myfile:
            myfile.write("\n" + str(format(trainReport['macro avg']['precision'], '.2f')) + "," + str(
                format(trainReport['macro avg']['recall'], '.2f'))
                         + "," + str(format(trainReport['macro avg']['f1-score'], '.2f')) + "," + str(
                format(testReport['macro avg']['precision'], '.2f'))
                         + "," + str(format(testReport['macro avg']['recall'], '.2f')) + "," + str(
                format(testReport['macro avg']['f1-score'], '.2f')) + "," + time.ctime())

trainLab0Score = []
trainLab1Score = []

for matrix in train_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

testLab0Score = []
testLab1Score = []

for matrix in test_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    testLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    testLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

parameters = []

for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag']:
    for C in [0.01, 0.05, 0.1, 1.0, 5.0, 10.0]:
        dictonary = {'solver': solver, 'C': C}
        parameters.append(dictonary)

Acc_Table = pd.DataFrame({"Parameters": parameters, "Train Accuracy": train_acc, "Train Label 0 Scores": trainLab0Score,
                          "Train label 1 Scores": trainLab1Score, "Train Avg Scores": train_preREf1,
                          "Test Accuracy": test_acc,
                          "Test Label 0 Scores": testLab0Score, "Test Label 1 Scores": testLab1Score,
                          "Test Avg Scores": test_preREf1}).sort_values(['Test Accuracy'], ascending=False)
Acc_Table.to_csv(file_name+"LR.csv", index=False)

"""## Loop for finding Hyper Parameters of AdaBoost """

with open("logs"+file_name+".csv", "a") as myfile:
    myfile.write("\n"+file_name+"_ADA")

cross_validation_scores = []  # Cross Validation Score
train_confus_matrix = []  # Confusion Matrix
test_confus_matrix = []  # Confusion Matrix

train_preREf1 = []
test_preREf1 = []

train_acc = []
test_acc = []  # Test Score

i = 0
for n_estimators in [50, 100, 200]:
    for base_estimator in [DecisionTreeClassifier(), ExtraTreeClassifier(), RandomForestClassifier(),
                           LogisticRegression()]:
        ada = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=42)
        ada.fit(X_train, Y_train)

        # cross_validation = cross_val_score(rfc,X_train,Y_train,scoring="accuracy",cv=5)
        # cross_validation_scores.append(cross_validation.mean())
        i += 1
        print(i)
        predictions = ada.predict(X_train)

        trainReport = classification_report(Y_train.values, predictions, output_dict=True)
        testReport = classification_report(Y_test.values, ada.predict(X_test), output_dict=True)
        train_preREf1.append({'precision': format(trainReport['macro avg']['precision'], '.2f'),
                              'recall': format(trainReport['macro avg']['recall'], '.2f'),
                              'f1Score': format(trainReport['macro avg']['f1-score'], '.2f')})
        test_preREf1.append({'precision': format(testReport['macro avg']['precision'], '.2f'),
                             'recall': format(testReport['macro avg']['recall'], '.2f'),
                             'f1Score': format(testReport['macro avg']['f1-score'], '.2f')})

        train_confus_matrix.append(confusion_matrix(Y_train.values, predictions))
        test_confus_matrix.append(confusion_matrix(Y_test.values, ada.predict(X_test)))

        with open("logs"+file_name+".csv", "a") as myfile:
            myfile.write("\n" + str(format(trainReport['macro avg']['precision'], '.2f')) + "," + str(
                format(trainReport['macro avg']['recall'], '.2f'))
                         + "," + str(format(trainReport['macro avg']['f1-score'], '.2f')) + "," + str(
                format(testReport['macro avg']['precision'], '.2f'))
                         + "," + str(format(testReport['macro avg']['recall'], '.2f')) + "," + str(
                format(testReport['macro avg']['f1-score'], '.2f')) + "," + time.ctime())

        train_acc.append(ada.score(X_train, Y_train))
        test_acc.append(ada.score(X_test, Y_test))

trainLab0Score = []
trainLab1Score = []

for matrix in train_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

testLab0Score = []
testLab1Score = []

for matrix in test_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    testLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    testLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

parameters = []

for n_estimators in [50, 100, 200]:
    for base_estimator in [DecisionTreeClassifier(), ExtraTreeClassifier(), RandomForestClassifier(),
                           LogisticRegression()]:
        dictonary = {'n_estimators': n_estimators, 'base_estimator': base_estimator}
        parameters.append(dictonary)

Acc_Table = pd.DataFrame({"Parameters": parameters, "Train Accuracy": train_acc, "Train Label 0 Scores": trainLab0Score,
                          "Train label 1 Scores": trainLab1Score, "Train Avg Scores": train_preREf1,
                          "Test Accuracy": test_acc,
                          "Test Label 0 Scores": testLab0Score, "Test Label 1 Scores": testLab1Score,
                          "Test Avg Scores": test_preREf1}).sort_values(['Test Accuracy'], ascending=False)
Acc_Table.to_csv(file_name + "ada.csv", index=False)


"""## Loop for finding Hyper Parameters of Gradient Boost"""

with open("logs"+file_name+".csv", "a") as myfile:
    myfile.write("\n"+file_name+"_GB")

cross_validation_scores = []  # Cross Validation Score
train_confus_matrix = []  # Confusion Matrix
test_confus_matrix = []  # Confusion Matrix

train_preREf1 = []
test_preREf1 = []

train_acc = []
test_acc = []  # Test Score

i = 0

for boosting_type in ['gbdt', 'dart', 'goss']:
    for n_estimators in [100, 200, 400]:
        for max_depth in [2, 3, 5]:
            gb = LGBMClassifier(class_weight='balanced', max_depth=max_depth, n_estimators=n_estimators,
                                boosting_type=boosting_type, random_state=42)

            gb.fit(X_train, Y_train)
            # cross_validation = cross_val_score(rfc,X_train,Y_train,scoring="accuracy",cv=5)
            # cross_validation_scores.append(cross_validation.mean())
            i += 1
            print(i)
            predictions = gb.predict(X_train)

            trainReport = classification_report(Y_train.values, predictions, output_dict=True)
            testReport = classification_report(Y_test.values, gb.predict(X_test), output_dict=True)
            train_preREf1.append({'precision': format(trainReport['macro avg']['precision'], '.2f'),
                                  'recall': format(trainReport['macro avg']['recall'], '.2f'),
                                  'f1Score': format(trainReport['macro avg']['f1-score'], '.2f')})
            test_preREf1.append({'precision': format(testReport['macro avg']['precision'], '.2f'),
                                 'recall': format(testReport['macro avg']['recall'], '.2f'),
                                 'f1Score': format(testReport['macro avg']['f1-score'], '.2f')})

            train_confus_matrix.append(confusion_matrix(Y_train.values, predictions))
            test_confus_matrix.append(confusion_matrix(Y_test.values, gb.predict(X_test)))

            train_acc.append(gb.score(X_train, Y_train))
            test_acc.append(gb.score(X_test, Y_test))

            with open("logs"+file_name+".csv", "a") as myfile:
                myfile.write("\n" + str(format(trainReport['macro avg']['precision'], '.2f')) + "," + str(
                    format(trainReport['macro avg']['recall'], '.2f'))
                             + "," + str(format(trainReport['macro avg']['f1-score'], '.2f')) + "," + str(
                    format(testReport['macro avg']['precision'], '.2f'))
                             + "," + str(format(testReport['macro avg']['recall'], '.2f')) + "," + str(
                    format(testReport['macro avg']['f1-score'], '.2f')) + "," + time.ctime())

trainLab0Score = []
trainLab1Score = []

for matrix in train_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    trainLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

testLab0Score = []
testLab1Score = []

for matrix in test_confus_matrix:
    pre = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    rec = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    f1 = 2 * pre * rec / (rec + pre)
    testLab0Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

    pre = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    rec = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    f1 = 2 * pre * rec / (rec + pre)
    testLab1Score.append({'precision': format(pre, '.2f'), 'recall': format(rec, '.2f'), 'f1Score': format(f1, '.2f')})

parameters = []

for boosting_type in ['gbdt', 'dart', 'goss']:
    for n_estimators in [100, 200, 400]:
        for max_depth in [2, 3, 5]:
            dictonary = {'boosting_type': boosting_type, 'boosting_type': boosting_type, 'max_depth': max_depth}
            parameters.append(dictonary)

Acc_Table = pd.DataFrame({"Parameters": parameters, "Train Accuracy": train_acc, "Train Label 0 Scores": trainLab0Score,
                          "Train label 1 Scores": trainLab1Score, "Train Avg Scores": train_preREf1,
                          "Test Accuracy": test_acc,
                          "Test Label 0 Scores": testLab0Score, "Test Label 1 Scores": testLab1Score,
                          "Test Avg Scores": test_preREf1}).sort_values(['Test Accuracy'], ascending=False)
Acc_Table.to_csv(file_name + "GB.csv", index=False)
