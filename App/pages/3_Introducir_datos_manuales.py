import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import pickle
import requests

st.set_page_config(page_title="Introducir datos manuales", page_icon=":pencil:")

st.subheader("Introduzca los datos de su transacción aquí:")


# Dentro del contexto del formulario
with st.form('Entrada manual de datos'):
    # Selección del día de la transacción
    day = st.slider('Día de la transacción', 1, 31)

    # Caja de entrada para números
    hour = st.slider('Hora de la transacción', 1, 24)

    # Caja de entrada para cantidad
    amount = st.number_input('Cantidad de la transacción', min_value=1)

    # Caja de entrada para retirada de fondos
    type_CASH_OUT = st.radio("¿Es una retirada de fondos?", ('Sí', 'No'))

    # Caja de entrada para transferencias
    type_TRANSFER = st.radio("¿Es una transferencia?", ('Sí', 'No'))

    # Caja de entrada para quien inicia la transacción
    nameorig_C = st.radio("¿Quién origina la transacción?", ('Cliente', 'Comercio'))

    # Caja de entrada para beneficiario
    namedest_C = st.radio("¿El beneficiario de la transacción es un cliente?", ('Sí', 'No'))

    # Caja de entrada para beneficiario
    namedest_M = st.radio("¿El beneficiario de la transacción es un comercio?", ('Sí', 'No'))

    submitted = st.form_submit_button("Submit")


if submitted:
    # Crear un diccionario con los datos de entrada
    data_dict = {
        'day': [day],
        'hour': [hour],
        'amount': [amount],
        'type_CASH_OUT': [1] if type_CASH_OUT == 'Sí' else [0],
        'type_TRANSFER': [1] if type_TRANSFER == 'Sí' else [0], 
        'nameorig_C': [1] if nameorig_C == 'Cliente' else [0],
        'namedest_C': [1] if namedest_C == 'Sí' else [0],
        'namedest_M': [1] if namedest_M == 'Sí' else [0]
    }

    # Crear el DataFrame a partir del diccionario
    X_test = pd.DataFrame(data_dict)

    # Imprimir los datos de prueba
    st.write("Vista previa de sus respuestas:")
    st.write(X_test)

    # 2. Realizar transformaciones de los datos
    # Cargamos el PCA y el scaler
    pca_url = 'https://github.com/LidiaMiranda/Fraud-detection-ML/raw/main/modelos/pca_5.pkl'
    scaler_url = 'https://github.com/LidiaMiranda/Fraud-detection-ML/raw/main/modelos/scaler.pkl'

    # Escalamos
    # Descargar el archivo scaler usando requests
    response_scaler = requests.get(scaler_url)
    # Guardar el contenido descargado en un archivo local
    with open('scaler.pkl', 'wb') as file:
        file.write(response_scaler.content)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    scaled_X_test = scaler.transform(X_test)

    # PCA
    # Descargar el archivo pca usando requests
    response_pca = requests.get(pca_url)
    # Guardar el contenido descargado en un archivo local
    with open('pca_5.pkl', 'wb') as file:
        file.write(response_pca.content)
    # Cargar el archivo pca desde el archivo local
    with open('pca_5.pkl', 'rb') as file:
        pca = pickle.load(file)

    X_test_pca = pd.DataFrame(pca.transform(scaled_X_test), columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

    # Descarga el contenido del archivo del modelo desde GitHub
    model_url = "https://github.com/LidiaMiranda/Fraud-detection-ML/raw/main/modelos/my_model.pkl"
    response = requests.get(model_url)

    # Verifica que la descarga haya sido exitosa
    if response.status_code == 200:
        # Carga el modelo desde el contenido descargado
        model = pickle.loads(response.content)
    else:
        # Si no se pudo descargar el modelo, muestra un mensaje de error
        raise Exception("No se pudo descargar el modelo desde GitHub.")

    # 4. Realizar predicción utilizando el modelo y los datos transformados
    y_pred = model.predict(X_test_pca)

    # Imprimir la predicción
    st.write("Predicción:")
    if y_pred == 1:
        st.write("Es una transacción segura")
    else:
        st.write("Es una transacción fraudulenta")


# mirar poner la fiabilidad el modelo, pero el modelo en sí, no este dato. 