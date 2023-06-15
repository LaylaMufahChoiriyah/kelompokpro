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

        st.info( " ## Nama Kelompok : ")
        st.write(" - Layla Mufah Choiriyah - 200411100052")
        st.write("- Diah Kamalia - 200411100061")
        st.info("## Repository Github")
        repo = "https://github.com/LaylaMufahChoiriyah/kelompokpro"
        st.markdown(f'[ Link Repository Github ]({repo})')
        st.info("## Link Colaboratory")
        repo1 = "https://colab.research.google.com/drive/1rQpD_c6EobkvysD6biIa-DGLrbRiiErQ?usp=sharing"
        st.markdown(f'[ Link Colaboratory ]({repo1})')
        
       
    elif choose == "Project":
        st.title("PSD B - Telkomsel Finance")
        data, preprocessing, model, implementasi = st.tabs(["Data","Preprocessing", "Model", "Implementasi"])
        with data:
            st.write("""# Load Dataset""")
            df = pd.read_csv("https://raw.githubusercontent.com/diahkamalia/DataMining1/main/TLKM.JK.csv")
            df
            sumdata = len(df)
            st.success(f"#### Total Data : {sumdata}")
            st.write("## Dataset Explanation")
            st.write("Data didapat dari :")
            repo2 = "https://finance.yahoo.com/"
            st.markdown(f'[ Yahoo Finance ]({repo2})')
            col1,col2 = st.columns(2)
            with col1:
                st.info("#### Data Type")
                df.dtypes
            with col2:
                st.info("#### Empty Data")
                st.write(df.isnull().sum())
            
            
                #===================================
             
                
                
        with preprocessing : 
            st.write("""# Preprocessing""")
            # transform univariate time series to supervised learning problem
            from numpy import array
            # split a univariate sequence into samples
            def split_sequence(sequence, n_steps):
                X, y = list(), list()
                for i in range(len(sequence)):
                # find the end of this pattern
                    end_ix = i + n_steps
                # check if we are beyond the sequence
                    if end_ix > len(sequence)-1:
                        break
                # gather input and output parts of the pattern
                    # print(i, end_ix)
                    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                    X.append(seq_x)
                    y.append(seq_y)
                return array(X), array(y)
            n_steps = 5
            X, y = split_sequence(data['Open'], n_steps)
            X.shape, y.shape
            st.info("#### Normalisasi ")
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(X)
            scaled

        with model : 
            st.write("""# Model""")
            st.info("## K - Nearest Neighbor")

        with implementasi:
            st.write("# Implementation")
            st.write("### Add Review :")
