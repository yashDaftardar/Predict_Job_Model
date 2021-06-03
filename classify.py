#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:58:38 2021

@author: hyadav
"""

import pandas as pd
import numpy as np
import re
import time
from string import punctuation
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import naive_bayes


from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def loadData(df):
    #split the data into 80:20; train-0.8 & test-0.2
    split = np.random.rand(len(df)) < 0.80
    train = df[split]
    test = df[~split]
    return train, test

def convert_utf8(s):
    return str(s)

def remove_urls(s):
    s = re.sub('[^\s]*.com[^\s]*', "", s)
    s = re.sub('[^\s]*www.[^\s]*', "", s)
    s = re.sub('[^\s]*.co.uk[^\s]*', "", s)
    return s

def remove_star_words(s):
    return re.sub('[^\s]*[\*]+[^\s]*', "", s)

def remove_nums(s):
    return re.sub('[^\s]*[0-9]+[^\s]*', "", s)

def remove_stopwords(s):
    global en_stopwords
    en_stopwords = stopwords.words('english')
    s = word_tokenize(s)
    s = " ".join([w for w in s if w not in en_stopwords])
    return s

def remove_punctuation(s):
    global punctuation
    for p in punctuation:
        s = s.replace(p, '')
    return s

def remove_hexcode(s):
    return re.sub('[^\s]*[^\x00-\x7f]+[^\s]*', "", s) 

def cleaning(dataset):
    
    #Assign column names
    dataset.columns =['Job_Description', 'Title']
    
    ##Data cleaning:
    dataset['Clean_Job_Description'] =dataset['Job_Description'].map(convert_utf8)
    dataset['Clean_Job_Description'] = dataset['Clean_Job_Description'].map(remove_urls)
    dataset['Clean_Job_Description'] = dataset['Clean_Job_Description'].map(lambda x: x.lower())
    dataset['Clean_Job_Description'] = dataset['Clean_Job_Description'].map(remove_star_words)
    dataset['Clean_Job_Description'] = dataset['Clean_Job_Description'].map(remove_nums)
    
    # Create a new column of descriptions with no stopwords
    dataset['Clean_Job_Description'] = dataset['Clean_Job_Description'].map(remove_stopwords)
    dataset['Clean_Job_Description'] = dataset['Clean_Job_Description'].map(remove_punctuation)
    dataset['Clean_Job_Description'] = dataset['Clean_Job_Description'].map(remove_hexcode)

    #remove punctuation and tokenize
    dataset["Tokens"] = dataset.apply(lambda row: word_tokenize(row['Clean_Job_Description']), axis=1)
    # #remove stopwords
    dataset['Tokens_1'] = dataset['Tokens'].apply(lambda x: [item for item in x if item not in stopwords.words('english')])
    # #merge tokens back into string text
    dataset['Text_3']=[" ".join(txt) for txt in dataset["Tokens_1"].values]
    # #create bigrams
    #dataset["Tokens_2"] = dataset["Tokens_1"].apply(lambda row: list(ngrams(row, 2)))

    return dataset

def traintestsplit(dataset):
    
    train_X, test_X, train_y, test_y = train_test_split(dataset['Text_3'],dataset['Title'],test_size=0.3)
    
    return train_X, test_X, train_y, test_y

def tfidf(dataset):
    
    #TF_IDF    
    tfidf_vect = TfidfVectorizer(max_features=500)
    tfidf_vect.fit(dataset["Text_3"])
    
    return tfidf_vect

def svmclassifier(train_X_tfidf, train_y, test_X_tfidf, test_y):    
    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_X_tfidf,train_y)

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(test_X_tfidf)
    
    # Use accuracy_score function to get the SVM accuracy
    # print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, test_y)*100)    
    # print("Area under the ROC curve:", roc_auc_score(test_y, predictions_SVM)*100)
    # print("Confusion Matrix:\n",confusion_matrix(test_y, predictions_SVM))

    return SVM

def naivebayesclassifier(train_X_tfidf, train_y, test_X_tfidf, test_y):
    
    # Classifier - Algorithm - Naive Bayes
    # fit the training dataset on the NB classifier
    Naive = naive_bayes.MultinomialNB()
    Naive.fit(train_X_tfidf,train_y)

    # predict the labels on validation dataset
    predictions_NB = Naive.predict(test_X_tfidf)

    # Use accuracy_score function to get the accuracy
    # print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, test_y)*100)
    # print("Area under the ROC curve:", roc_auc_score(test_y, predictions_NB))
    # print("Confusion Matrix:\n",confusion_matrix(test_y, predictions_NB))
    
    return Naive

def logisticregressionclassifier(train_X_tfidf, train_y, test_X_tfidf, test_y):
    # Classifier - Algorithm - LogisiticRegression
    # fit the training dataset on the classifier
    Log_classifier = LogisticRegression(max_iter=10000)
    Log_grid = [{'C': [0.5, 1, 1.5], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}]
    gridsearchLogisticreg = GridSearchCV(Log_classifier, Log_grid, cv=3)
    gridsearchLogisticreg.fit(train_X_tfidf, train_y)

    # predict the labels on validation dataset
    predictions_lr = gridsearchLogisticreg.predict(test_X_tfidf)
    
    # Use accuracy_score function to get the SVM accuracy
    # print('Accuracy of the model: ', accuracy_score(predictions_lr, test_y)*100)
    # print("Area under the ROC curve:", roc_auc_score(test_y, predictions_lr, multi_class='ovo'))
    # print("Confusion Matrix:\n",confusion_matrix(test_y, predictions_lr))
    
    return gridsearchLogisticreg

if __name__ == "__main__":

    start_time = time.time()
    
    #Load the train dataset
    df = pd.read_csv("Job_Ads.csv")
    
    # Remove duplicates
    df2 = df.drop_duplicates()

    #Cleaning
    dataset = cleaning(df2)

    #Split the dataset into train and test
    train_X, test_X, train_y, test_y = traintestsplit(dataset)

    # #target variable label encoding
    # Encoder = LabelEncoder()
    # train_y = Encoder.fit_transform(train_y)
    # test_y = Encoder.fit_transform(test_y)

    #Featurizing: TFIDF Vectorizer
    tfidf_vect = tfidf(dataset)

    train_X_tfidf = tfidf_vect.transform(train_X)
    test_X_tfidf = tfidf_vect.transform(test_X)
    
    #Models
    SVM_model = svmclassifier(train_X_tfidf, train_y, test_X_tfidf, test_y)
    # print("--- %s seconds ---" % (time.time() - start_time))    
    
    NB_model = naivebayesclassifier(train_X_tfidf, train_y, test_X_tfidf, test_y)
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    LR_model = logisticregressionclassifier(train_X_tfidf, train_y, test_X_tfidf, test_y)
    # print("--- %s seconds ---" % (time.time() - start_time))    
    
    # #Predict
    # test_predictions_SVM = SVM_model.predict(test_X_tfidf)
    # print('Accuracy of the model: ', accuracy_score(test_predictions_SVM, test_y)*100)

    # test_predictions_NB = NB_model.predict(test_X_tfidf)
    # print('Accuracy of the model: ', accuracy_score(test_predictions_NB, test_y)*100)

    # test_predictions_LR = LR_model.predict(test_X_tfidf)
    # print('Accuracy of the model: ', accuracy_score(test_predictions_LR, test_y)*100)

    
    #Load the test prediction dataset
    df_test_prediction = pd.read_csv("TEST.csv")
    
    #Cleaning
    df_test_prediction = cleaning(df_test_prediction)
    
    #Featurizing: TFIDF Vectorizer
    test_prediction_tfidf = tfidf(df_test_prediction)
    test_prediction_X_tfidf = tfidf_vect.transform(df_test_prediction['Job_Description'])
    
    #Predict
    # test_predictions_SVM = SVM_model.predict(test_prediction_X_tfidf)
    # print('Accuracy of the model: ', accuracy_score(test_predictions_SVM, df_test_prediction['Title'])*100)

    # test_predictions_NB = NB_model.predict(test_prediction_X_tfidf)
    # print('Accuracy of the model: ', accuracy_score(test_predictions_NB, df_test_prediction['Title'])*100)

    test_predictions_LR = LR_model.predict(test_prediction_X_tfidf)
    # print('Accuracy of the model: ', accuracy_score(test_predictions_LR, df_test_prediction['Title'])*100)
    
    
    #  a new file that includes the predicted label for each line in the test file.
    # df_test_prediction['Jobrole'] = test_predictions_SVM
    # df_test_prediction['Job_Description','Jobrole'].to_csv('Real Test SVM.csv', index=False)

    # df_test_prediction['Jobrole'] = test_predictions_NB
    # df_test_prediction['Job_Description','Jobrole'].to_csv('Real Test NB.csv', index=False)
    
    df_test_prediction['Predicted Title'] = test_predictions_LR
    df_test_prediction[['Job_Description','Predicted Title']].to_csv('Prediction.csv', index=False, header=False)
    
    