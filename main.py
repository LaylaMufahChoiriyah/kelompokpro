import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.validation import joblib
import joblib
from PIL import Image
import io

from streamlit_option_menu import option_menu
st.set_page_config(page_title="Proyek Sains Data", page_icon='')

with st.container():
    with st.sidebar:
        choose = option_menu("Telkomsel Finance", ["Home", "Project"],
                             icons=['house', 'basket-fill'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "10A19D"},
            "icon": {"color": "#fb6f92", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#c6e2e9"},
            "nav-link-selected": {"background-color": "#a7bed3"},
        }
        )

    if choose == "Home":
        
        st.markdown("# Proyek Sains Data B")

        st.info( " Nama Kelompok : ")
        st.write(" - Layla Mufah Choiriyah - 200411100052")
        st.write("- Diah Kamalia - 200411100061")
        st.write("## Repository Github")
        st.write(" Click the link below to access the source code")
        repo = "https://github.com/LaylaMufahChoiriyah/kelompokpro"
        st.markdown(f'[ Link Repository Github ]({repo})')
        st.write("## Link Colaboratory")
        repo1 = "https://colab.research.google.com/drive/1rQpD_c6EobkvysD6biIa-DGLrbRiiErQ?usp=sharing"
        st.markdown(f'[ Link Colaboratory ]({repo1})')
        
       
    elif choose == "Project":
        st.title("Data Mining - Water Quality Prediction")
        st.write("Diah Kamalia - 200411100061")
        desc, dataset, preprocessing, modelling, implementation = st.tabs(["Deskripsi Data", "Dataset", "Preprocessing", "Modelling", "Implementasi"])
        
