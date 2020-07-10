import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import io

def main():
    load_model()
    st.title('Nub 1.0 Summarizer')
    input_text = st.text_area("Enter text")
    'or...'
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    if st.button('Summarize'):
        if uploaded_file and not input_text:
            text_from_file = process_file(uploaded_file)
            run_summarizer(text_from_file)
        else:
            if input_text:
                run_summarizer(input_text)
            else:
                'gotta give *some* input'
    #todo: add radio button



def process_file(uploaded_file):
    input_text = uploaded_file.getvalue()

    # uploaded_file = io.StringIO(uploaded_file)
    # input_text = open(uploaded_file, "r")

    # Retrieve file contents -- this will be
    # 'First line.\nSecond line.\n'
    # input_text = uploaded_file.getvalue()
    # uploaded_file.close()
    # with open(uploaded_file, 'r') as file:
    #     input_text = file.read().replace('\n', '')

    return input_text

@st.cache
def load_model():
    pass

def run_summarizer(input_text):


    'brb...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i + 1}')
        bar.progress(i + 1)
        time.sleep(0.01)

    '...and here\'s the summary'
    st.write(input_text)
    bar.empty()




if __name__==main():
    main()