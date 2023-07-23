import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle


st.set_page_config(page_title="Home", page_icon=":house_with_garden:")
#---Opciones predeterminadas de la app---
# Establecer el tamaño máximo de carga a 500 MB (en bytes)
max_size = 500 * 1024 * 1024  # 500 MB en bytes

# Csv train ya procesado (PCA=5, Scaler)
df_train = pd.read_csv('C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/data/processed/df_red70_train_pca3.csv')
# Dividimos en X e y
X_train = df_train.drop(columns=['isfraud'])
y_train = df_train['isfraud']

#Cargamos modelo, pca y scaler

with open('C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/modelos/my_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Definir el título utilizando Markdown y agregar el estilo de CSS para centrar
st.markdown(
    "<h2 style='text-align: center;'>Detección de fraude en pagos con teléfonos móviles</h2>", 
    unsafe_allow_html=True
)

# Mostrar la imagen utilizando la URL directamente
st.image('https://d6xcmfyh68wv8.cloudfront.net/learn-content/uploads/2020/04/ecommerce-fraud-770x515.jpeg', use_column_width=True)
#Subtítulo
st.markdown(
    "<h3 style='text-align: center;'>Compruebe si sus transacciones son seguras gracias a nuestra herramienta</h3>",
    unsafe_allow_html=True
)
#Separación vertical
st.markdown("<br>", unsafe_allow_html=True)

st.write('👈Escoja de la **barra lateral** el método que utilizará para verificar sus datos.')
st.write('	❗**IMPORTANTE**: si desea subir un archivo .csv, no olvide utilizar nuestra **plantilla**.')