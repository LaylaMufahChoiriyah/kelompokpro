import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from numpy import array
import joblib
from PIL import Image
import io
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import datetime
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
            st.write("### Sumber Data :")
            st.info("Data yang kami gunakan merupakan data volume historical prices dari Perusahaan Perseroan (Persero) PT Telekomunikasi Indonesia Tbk (TLKM.JK) atau Telkomsel dari website Yahoo Finance. ")
            st.write("### Link Data :")
            repo2 = "https://finance.yahoo.com/"
            st.markdown(f'[ Yahoo Finance ]({repo2})')
            st.write("### Tipe data :")
            st.info("Tipe data yang digunakan yaitu time series dari volume saham yang diambil dari rentang waktu 15 Juni 2022 sampai 15 Juni 2023. ")
            st.write("### Tentang Data :")
            st.info("Data yang digunakan merupakan data volume dari saham Perusahaan Perseroan (Persero) PT Telekomunikasi Indonesia Tbk (TLKM.JK) atau Telkomsel, yang mana data tersebut merupakan data histori harga dengan frekuensi harian yang diambil dari rentang waktu 15 Juni 2022 - 15 Juni 2023.")

            col1,col2 = st.columns(2)
            with col1:
                st.info("#### Data Type")
                df.dtypes
            with col2:
                st.info("#### Empty Data")
                st.write(df.isnull().sum())
            
             
            
        with preprocessing : 
            st.write("""# Preprocessing""")
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
            X, y = split_sequence(df['Open'], n_steps)
            st.info("## Ukuran Data dan Target ")
            st.warning("X")
            X.shape
            st.warning("Y")
            y.shape
            st.info("## Normalisasi ")
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(X)
            scaled
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(scaled, y, test_size=0.2, random_state=0, shuffle=False)
            st.info("## Split Data ")
            st.write("### Data Test")
            X
            st.write("### Data Training")
            y
            pipeline = Pipeline([
                ('pca', PCA(n_components=4)),
                ('algo', GaussianNB())
            ])
            
            model_naive_3 = RandomizedSearchCV(pipeline, {}, cv=4, n_iter=50, n_jobs=-1, verbose=1, random_state=42)
            model_naive_3.fit(X_train, y_train)
            
            st.write(f'Parameter Terbaik: {model_naive_3.best_params_}')
            st.write(model_naive_3.score(X_train, y_train), model_naive_3.best_score_, model_naive_3.score(X_test, y_test))

        with model : 
            st.write("""# Model""")
            st.info("## Naïve Bayes")
            # Split Data
            
            # Training
            gNB = GaussianNB()
            gNB.fit(X_train, y_train)
            y_pred=gNB.predict(X_train)
            st.write("## Akurasi :")
            st.info(f'Akurasi dari Mean Absolute Percentage Error adalah = {mean_absolute_percentage_error(y_train, y_pred)}')
        with implementasi:
            st.write("# Implementation")
            st.write("### Input Data :")
            Date = st.date_input("Date",datetime.date(2019, 7, 6))
            Open = st.number_input("Open")
            High = st.number_input("High")
            Low = st.number_input("Low" )
            Close = st.number_input("Close")
            Adj_Close = st.number_input("Adj Close")
            Volume = st.number_input("Volume")
            result = st.button("Submit")
            input = [[Date, Open, High, Low, Close, Adj_Close, Volume]]
                input_norm = scaler.transform(input)
                FIRST_IDX = 0
                use_model = gNB
                predictresult = use_model.predict(input)[FIRST_IDX]
                st.write("## Akurasi :")
                st.info(f'Akurasi dari Mean Absolute Percentage Error adalah = {mean_absolute_percentage_error(y_train, y_pred)}')

                
