from collections import deque

import streamlit as st
import pickle
import pandas as pd
import regex as re
from streamlit_option_menu import option_menu
import math


# install streamlit options menu for side bar
# install pickle5 to load and run model

# Page Information Classes
class PredictionPage:
    def __init__(self, user_inputs, title, model):
        self.user_inputs = user_inputs
        self.title = title
        self.model = model


def get_user_input(dataset):
    df = pd.read_csv(dataset, nrows=0)
    list(df.columns)
    if 'name' in df.columns:
        df.drop('name', axis=1, inplace=True)
    if 'Outcome' in df.columns:
        df.drop('Outcome', axis=1, inplace=True)
    return [re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', label).capitalize() for label in
            list(df.columns)]


# loading saved models
diabetes_model = pickle.load(open('saved models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('saved models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('saved models/parkinsons_model.sav', 'rb'))

# create page list
diabetes_page = PredictionPage(get_user_input('dataset/diabetes.csv'), 'Diabetes', diabetes_model)
heart_disease_page = PredictionPage(get_user_input('dataset/heart.csv'), 'Heart Disease', heart_model)
parkinsons_page = PredictionPage(get_user_input('dataset/parkinsons.csv'), 'Parkinsons', parkinsons_model)

# page dictionary
pages = {diabetes_page.title: diabetes_page,
         heart_disease_page.title: heart_disease_page,
         parkinsons_page.title: parkinsons_page}

# data set list
datasets = ['dataset/diabetes.csv', 'dataset/heart.csv', 'dataset/parkinsons.csv']

# sidebar for navigation (streamlit)
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           list(pages.keys()),
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Page Creation Function
def create_page(page: PredictionPage):
    st.title(page.title + ' Prediction')
    model_args = []

    with st.expander('Layout Options'):
        num_cols = st.slider('Number of Columns to Display User Input', 1, len(page.user_inputs))
    num_rows = math.ceil(len(page.user_inputs) / num_cols)  # automate rows and columns
    user_input_queue = deque(page.user_inputs)
    for row in range(0, num_rows):
        cols = st.columns(num_cols)
        for column_index in range(0, num_cols):
            if len(user_input_queue) > 0:
                model_args.append(cols[column_index].text_input(user_input_queue.popleft()))

    if st.button('Predict'):
        prediction_value = page.model.predict([model_args])
        st.success(page.title + ' Not Present' if prediction_value[0] == 0
                   else page.title + ' Present')


create_page(pages.get(selected))
