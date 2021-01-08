import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report

def train_model(model_name, model, X_train, X_test, y_train, y_test):
    print(f'BEGIN. {model_name.upper()}......')
    model.fit(X_train, y_train)
    pickle.dump(model, open(f"models/{model_name.replace(' ', '_')}.sav", 'wb'))
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    print(f'TESTING DATA----> {model_name.upper()}: \t\t{accuracy_score(y_test, y_pred) * 100:.2f}%')
    print(f'TRAINING DATA---> {model_name.upper()}: \t\t{accuracy_score(y_train, y_train_pred) * 100:.2f}%')
    print(classification_report(y_test, y_pred))
    print(f'END. {model_name.upper()}')
    print('======================================================')
    return y_pred

data = pd.read_csv('data/train.csv')

print('=============Splitting the data=============')
X = data.text
y = data.target
print(f'Data shape: {data.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'X_Train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_Test shape: {X_test.shape}, y_test shape: {y_test.shape}')

print('\n============Message Preprocessing============')
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2, stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

print('Training and testing data shape after pre-processing:')
print(f'X_Train shape: {X_train_vect.shape}, y_train shape: {y_train.shape}')
print(f'X_Test shape: {X_test_vect.shape}, y_test shape: {y_test.shape}')

print('\n=============Model Building==================')
lr_model = LogisticRegression()
lr_y_pred = train_model('Logistic Regression', lr_model, X_train_vect, X_test_vect, y_train, y_test)

svm_model = LinearSVC()
svm_y_pred = train_model('Support Vector Machine', svm_model, X_train_vect, X_test_vect, y_train, y_test)

nb_model = MultinomialNB()
nb_y_pred = train_model('Naive Bayes', nb_model, X_train_vect, X_test_vect, y_train, y_test)
