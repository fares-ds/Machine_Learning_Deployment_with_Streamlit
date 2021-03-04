import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import sqlite3
import hashlib

def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_haches(password, hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

def add_userdata(username, password):
	c.execute('INSERT INTO userstable(username, password) VALUES (?, ?)', (username, password))
	conn.commit()

def login_user(username, password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password =?', (username, password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


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
		return 'Disaster'
	return 'Not Disaster'
def home():
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

def analytics():
	# Input Text In a box
	st.header('Enter Your tweet here: ')
	text_input = ''
	text = st.text_area('Text input', text_input, height=250)

	st.button('Make Prediction!')

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
	st.warning(lr_result)

	st.header('Support Vector Machine: ')
	st.warning(svm_result)

	st.header('Naive Bayes: ')
	st.warning(nb_result)

menu = ['Home', 'Login', 'SignUp']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
	st.subheader("Home")
	home()

elif choice == 'Login':
	st.subheader("Login Section")
	username = st.sidebar.text_input("User Name")
	password = st.sidebar.text_input("Password", type='password')
	if st.sidebar.checkbox("Login"):
		create_usertable()
		result = login_user(username, password)
		if result:
			st.success(f"Logged In as {username}")

			task = st.selectbox("Task", ["Analytics", "Profiles"])
			if task == "Analytics":
				st.subheader("Analytics")
				analytics()
			elif task == "Profiles":
				st.subheader("User Profiles")
				user_result = view_all_users()
				clean_db = pd.DataFrame(user_result, columns=['Username', 'Password'])
				st.dataframe(clean_db)
		else:
			st.warning("Incorrect Username ")
		
elif choice == 'SignUp':
	st.subheader("Create New Account")
	new_user = st.text_input("Username")
	new_password = st.text_input("Password", type='password')

	if st.button("SingUp"):
		create_usertable()
		add_userdata(new_user, new_password)
		st.success("You have successfully created a valid Account")
		st.info('Go to Login Menu to login')

