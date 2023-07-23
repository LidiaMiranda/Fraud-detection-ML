
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import pickle
import base64
import io

#Especificaciones del archivo (tamaño máximo)
max_size = 500 * 1024 * 1024  # 500 MB en bytes

# Csv train ya procesado
df_train = pd.read_csv('C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/data/processed/df_red70_train_pca3.csv')
# Dividimos en X e y
X_train = df_train.drop(columns=['isfraud'])
y_train = df_train['isfraud']

# Cargar el modelo (es de ejemplo)
model = pickle.load(open('C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/modelos/my_model.pkl', 'rb'))

#Icono de la pestaña
st.set_page_config(page_title="Cargar CSV", page_icon="	:outbox_tray:")

#Título
st.markdown(
    "<h3 style='text-align: center;'>Suba sus datos utilizando nuestra plantilla</h3>", 
    unsafe_allow_html=True
)
#Separación vertical
st.markdown("<br>", unsafe_allow_html=True)

# Botón para cargar archivo CSV en la primera columna

uploaded_file = st.file_uploader("Cargue su archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer los datos del archivo cargado en un búfer de bytes
    buffer = io.BytesIO(uploaded_file.read())

    # Verificar si el tamaño del archivo está dentro del límite
    file_size = len(buffer.getvalue())
    if file_size <= max_size:
        # Leer el archivo CSV desde el búfer de bytes y mostrar los datos
        test = pd.read_csv(buffer)
        # dividimos en X e y
        X_test = test.drop(columns=['isfraud'])
        y_test = test['isfraud']
        # importamos ruta de nuestro preprocesado guardado
        ruta_pca = 'C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/modelos/pca_5.pkl'
        ruta_scaler = 'C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/modelos/scaler.pkl'

        try:
            # Escalamos
            scaler = pickle.load(open(ruta_scaler, 'rb'))
            scaled_X_test = scaler.transform(X_test)

            # PCA
            pca = PCA(n_components=5)
            X_test_pca = pca.fit_transform(scaled_X_test)
            X_test_pca = pd.DataFrame(pca.transform(scaled_X_test), columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

            # Predicción
            predictions = model.predict(X_test_pca)

            # Agregar la columna de predicciones al DataFrame
            X_test['Predicción'] = predictions

            # Datos de etiquetas
            labels = ['No fraude', 'Fraude']
            values = X_test['Predicción'].value_counts()
            fraud = X_test[X_test['Predicción'] == 1]

            st.subheader('Resultados de la predicción')

            st.write(f'Transacciones fraudulentas: {values[1]}')
            st.write(f'Transacciones seguras: {values[0]}')

            #Separación vertical
            st.markdown("<br>", unsafe_allow_html=True)

            f1_score_value = f1_score(y_test, predictions, average='weighted')
            formatted_f1_score = round(f1_score_value, 3)
            st.write(f'F1-score: {formatted_f1_score}')

            # Crear el gráfico de donut
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
            fig.update_traces(hoverinfo='label+percent')
            fig.update_layout(title='Proporción de transacciones fraudulentas y seguras')
            st.plotly_chart(fig)

            # Mostrar los las transacciones fraudulentas
            st.write("Vista previa de las transacciones fraudulentas:")
            st.write(fraud.head())

            # Crear un objeto de tipo archivo en memoria
            csv_file = io.StringIO()
            fraud.to_csv(csv_file, sep=';', index=False)
            csv_file.seek(0)
            # Botón de descarga
            b64 = base64.b64encode(csv_file.getvalue().encode()).decode()  # Convertimos el contenido del CSV a binario base64
            csv_filename = "transacciones_fraudulentas.csv"
            csv_mime = "text/csv"
            st.markdown(
                f'<div style="display: flex; justify-content: center;">\
                <a download="{csv_filename}" href="data:{csv_mime};base64,{b64}" class="download-link">\
                <button type="button" style="cursor: pointer;">Descargar transacciones fraudulentas 📄</button>\
                </a>\
                </div>',
                unsafe_allow_html=True)


        except Exception as e:  
            st.write("Error al procesar los datos o hacer la predicción:", e)
    else:
        st.write("El tamaño del archivo supera el límite permitido de 500 MB.")


# # Comprobar si el usuario ha cargado un archivo
# if uploaded_file is not None:
#     # Leer los datos del archivo cargado en un búfer de bytes
#     buffer = BytesIO(uploaded_file.read())

#     # Verificar si el tamaño del archivo está dentro del límite
#     file_size = len(buffer.getvalue())
#     if file_size <= max_size:
#         # Leer el archivo CSV desde el búfer de bytes y mostrar los datos
#         test = pd.read_csv(buffer)
#     #dividimos en X e y
#         X_test = test.drop(columns=['isfraud'])
#         y_test = test['isfraud']
#     #importamos ruta de nuestro preprocesado guardado
#         ruta_pca = 'C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/modelos/pca_5.pkl'
#         ruta_scaler = 'C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/modelos/scaler.pkl'

#         try:
#             # Escalamos
#             scaler = pickle.load(open(ruta_scaler, 'rb'))
#             scaled_X_test = scaler.transform(X_test)

#             # PCA
#             pca = PCA(n_components=5)
#             X_test_pca = pca.fit_transform(scaled_X_test)
#             X_test_pca = pd.DataFrame(pca.transform(scaled_X_test), columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

#             # Predicción
#             predictions = model.predict(X_test_pca)

#             # # Mostrar los datos originales
#             # st.write("Vista previa de los datos cargados desde el archivo CSV:")
#             # st.write(X_test.head())

#             # Agregar la columna de predicciones al DataFrame
#             X_test['Predicción'] = predictions

#             # Datos de etiquetas
#             labels = [ 'No fraude','Fraude']
#             values = X_test['Predicción'].value_counts()

#             st.subheader('Resultados de la predicción')

#             st.write(f'Transacciones fraudulentas: {values[1]}')
#             st.write(f'Transacciones seguras: {values[0]}')

#             # Crear el gráfico de donut
#             fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
#             fig.update_traces(hoverinfo='label+percent')
#             fig.update_layout(title='Proporción de transacciones fraudulentas y seguras')

#             # Mostrar el gráfico en la primera columna
#             st.plotly_chart(fig)

#         except Exception as e:  
#             st.write("Error al procesar los datos o hacer la predicción:", e)
#     else:
#             st.write("El tamaño del archivo supera el límite permitido de 500 MB.")
#             #.