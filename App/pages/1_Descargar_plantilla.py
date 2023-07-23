import streamlit as st
from utils.utilsML import *


st.set_page_config(page_title="Descargar plantilla", page_icon=":inbox_tray:")

st.markdown(
    "<h2 style='text-align: center;'>Descargar plantilla</h2>",
    unsafe_allow_html=True
)
# Separación vertical
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
     "<h7 style='text-align: center;'>Descargue la plantilla en formato .csv que usará posteriormente para subir sus datos.</h7>",
    unsafe_allow_html=True
)
#Generar dataset vacío como plantilla
plantilla = pd.DataFrame(columns=['day', 'hour', 'amount', 'type_CASH_OUT', 'type_TRANSFER', 'nameorig_C', 'namedest_C', 'namedest_M'])
csv = plantilla.to_csv(index=False)

# Modificar formato del CSV con espacios
csv_file = io.StringIO()
plantilla.to_csv(csv_file, index=False, sep=";")  # Usamos ; como separador
csv_file.seek(0)

# Botón de descarga
b64 = base64.b64encode(csv_file.read().encode()).decode()  # Convertimos el CSV a binario base64
csv_filename = "plantilla.csv"
csv_mime = "text/csv"
st.markdown(
    f'<div style="display: flex; justify-content: center;">\
        <a download="{csv_filename}" href="data:{csv_mime};base64,{b64}" class="download-link">\
            <button type="button" style="cursor: pointer;">Descargar plantilla 📄</button>\
        </a>\
    </div>',
    unsafe_allow_html=True
)
# Separación vertical
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

#Vista previa plantilla
st.markdown(
    "<h5 style='text-align: left;'>Vista previa de la plantilla</h5>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(data=plantilla, use_container_width=True)

#Separación vertical
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

#Explicación columnas
st.markdown(
    "<h5 style='text-align: left;'>Instrucciones para rellenar la plantilla: </h5>",
    unsafe_allow_html=True
)
st.divider()

st.write("Columna '**day**'. Numérica. Refleja el día del mes en el que se hizo la transacción. Solo acepta números del 1 al 31.")

st.divider()

st.write("Columna '**hour**'. Numérica. Refleja la hora del día en el que se hizo la transacción. Solo acepta números del 1 al 24, sin reflejar los minutos.")

st.divider()

st.write("Columna '**amount**'. Numérica. Cantidad de la transacción. No acepta decimales.")

st.divider()

st.write("Columna '**type_CASH_OUT**'. Numérica. Si la transacción es una retirada de fondos, escriba 1. En caso contrario, escriba 0.")

st.divider()

st.write("Columna '**type_TRANFER**'. Numérica. Si la transacción es una transferencia, escriba 1. En caso contrario, escriba 0.")

st.divider()

st.write("Columna '**nameorig_C**'. Numérica. Si la transacción la inicia un cliente, escriba 1. En caso contrario, escriba 0.")

st.divider()

st.write("Columna '**namedest_C**'. Numérica. Si el beneficiario de la transacción es un cliente, escriba 1. En caso contrario, escriba 0.")

st.divider()

st.write("Columna '**namedest_M**'. Numérica. Si el beneficiario de la transacción es un comercio, escriba 1. En caso contrario, escriba 0.")

st.divider()