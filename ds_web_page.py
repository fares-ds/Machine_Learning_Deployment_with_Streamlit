import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# print('Uploading ML models')
lr_model = pickle.load(open('models/Logistic_Regression.sav', 'rb'))
svm_model = pickle.load(open('models/Support_Vector_Machine.sav', 'rb'))
nb_model = pickle.load(open('models/Naive_Bayes.sav', 'rb'))
# print('ML models upload successfully....')

# print('Uploading Tfidf vectorizer models')
tfidf = pickle.load(open('models/tfidf.sav', 'rb'))
# print('Tfidf vectorizer upload successfully....')

def predict_tweet(model, text):
	vectorize_text = tfidf.transform([text])
	target = model.predict(vectorize_text)[0]
	return target

def target_value(target):
	if target:
		return 'Disater'
	return 'Not Disaster'

st.title('Disaster Tweets Classification')
st.markdown("""
This app performs simple tweets classification into Disaster or Not
* **GitHub repo**: [Machine_Learning_Deployment_with_Streamlit](https://github.com/fares-ds/Machine_Learning_Deployment_with_Streamlit)
* **Data**: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data)
	""")
image = Image.open('images/data_analysis.png')
st.image(image, use_column_width=True)
st.markdown("""
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency 
they’re observing in real-time. Because of this, more agencies are interested in 
programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).	
But, it’s not always clear whether a person’s words are actually announcing a disaster.


This app performs simple classification of tweet into disaster or Not.
""")

# Input Text In a box
st.header('Enter Your tweet here: ')
text_input = ''
text = st.text_area('Text input', text_input, height=250)

lr_result = target_value(predict_tweet(lr_model, text))
svm_result = target_value(predict_tweet(svm_model, text))
nb_result = target_value(predict_tweet(nb_model, text))


st.write("""
***
	""")

# Prints the input text
st.header('INPUT TEXT: ')
text

# Text words count
st.header('Logistic Regression: ')
lr_result

st.header('Support Vector Machine: ')
svm_result

st.header('Naive Bayes: ')
nb_result
