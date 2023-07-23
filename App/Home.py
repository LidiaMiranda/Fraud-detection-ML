
import streamlit as st

st.set_page_config(page_title="Home", page_icon=":house_with_garden:")
#---Opciones predeterminadas de la app---
# Establecer el tamaño máximo de carga a 500 MB (en bytes)
max_size = 500 * 1024 * 1024  # 500 MB en bytes

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