import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from collections import Counter

st.title('Disaster Tweets Classification')
st.markdown("""
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency 
they’re observing in real-time. Because of this, more agencies are interested in 
programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).	
But, it’s not always clear whether a person’s words are actually announcing a disaster.


This app performs simple classification of tweet into disaster or Not.
""")

# Input Text In a box
st.header('Enter Your text: ')
text_input = '>Text Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT'

text = st.text_area('Text input', text_input, height=250)
text = text.splitlines()
text = text[1:]
text = ''.join(text)

st.write("""
***
	""")

# Prints the input text
st.header('INPUT TEXT: ')
text

# Text words count
st.header('OUTPUT (NUMBER OF WORDS): ')
words_count = len(text.split())
words_count

# Word counter
st.header('OUTPUT (WORDS COUNT)')
count = Counter()
count.update(text.split())
count
