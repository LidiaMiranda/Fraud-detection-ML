<div align="center">
  <h1>DETECCIÓN DE FRAUDE EN PAGOS CON TELÉFONOS MÓVILES</h1>
  <img src="https://aniview.com/wp-content/uploads/2022/02/mobile-ad-fraud.png" width="500"/>
</div>

## 1. TEMA

Pagar con el teléfono móvil ha incrementado notablemente su popularidad, ya que permite tanto realizar compras en comercios físicos y online como intercambiar dinero en pequeñas cantidades entre amistades, familia o particulares en general. Una app y un smartphone harán la labor de transferir dinero de una cuenta o tarjeta a otra persona o comercio.

Según datos del informe de 2020 que realizó Worldpay sobre pagos en todo el mundo, el uso de tarjetas de crédito y débito han bajado entre 2019 y 2020 de un 35% conjunto a un 27%. Por su parte, los pagos digitales o pagos móviles siguieron con su tendencia ascendente. De un 42% de uso en 2019 en todo el mundo a un 52% en 2020. Y todo apunta a que seguirá siendo así.

La pregunta que surge por parte de firmas de seguridad es: ¿es el pago móvil seguro? ¿Deberíamos adquirir nuevos hábitos de seguridad para no caer en estafas online? La popularidad del pago móvil lo coloca en la diana del fraude online. Así que es inevitable que las estafas dirigidas al usuario doméstico se hagan pasar por métodos de pago móvil si cada vez es más popular.

El objetivo del proyecto crear un modelo de Machine Learning que permita detectar si una transacción realizada con teléfono móvil es fraudulenta.

## 2. DATASET ORIGINAL Y ACONDICIONAMIENTO DE DATOS

### 2.1. Dataset

- Nombre: Synthetic Financial Datasets For Fraud Detection
- URL: https://www.kaggle.com/datasets/ealaxi/paysim1
- Breve resumen del contenido del DataSet:

Debido a la dificultad de encontrar datasets de transacciones bancarias con datos reales (por políticas de privacidad) que ayuden a generar modelos de prevención de fraude, se ha utilizado la herramienta **PaySim** para generar este dataset.

PaySim es un software que automatiza las pruebas de sistemas de pago. Este tipo de software puede simular una gran cantidad de escenarios de pago, permitiendo a bancos y empresas financieras evaluar sus sistemas de forma concienzuda y asegurar que están funcionando correctamente.

Por lo tanto, es una herramienta especializada en generar datos ficticios sobre transacciones de pago para facilitar la labor de quienes deseean generar herramientas de prevención de fraude, entre otras.

#### 2.1.1 Explicación columnas

A continuación se resumirá brevemente el contenido de las columnas:

- Step. Numérica. Representa la hora del mes en el que se hace la transacción.

- Type. Categórica. Tipo de transacción (pago, transferencia, retirada de fondo, etc.).

- Amount. Numérica. Importe de la transacción.

- NameOrig. Categórica. Código de indentificación de quien origina la transacción. Consta de una letra C o M seguida de una serie de números. C significa "customer" y M significa "merchant".

- OldbalanceOrg. Numérica. Representa el balance en cuenta de quien origina la transacción antes de que se realice la misma.

- NewbalanceOrig. Numérica. Representa el balance en cuenta de quien origina la transacción después de que se realice la misma.

- NameDest. Categórica. Código de indentificación del beneficiario de la transacción. Consta de una letra C o M seguida de una serie de números. C significa "customer" y M significa "merchant".

- OldbalanceDest. Numérica. Representa el balance en cuenta del beneficiario de la transacción antes de que se realice la misma.

- NewbalanceDest.Numérica. Representa el balance en cuenta del beneficiario de la transacción después de que se realice la misma.

- IsFraud.Numérica. Representa si la transacción es fraudulenta (1) o no (0).

- IsFlaggedFraud. Representa si la transacción es sospechosa de fraude (1) o no (0).

## 3. APP STREAMLIT

Este proyecto tiene asociado una app hecha con Streamlit. Toda la información sobre el código se encuentra en la carpeta App. 

[https://fraud-detection-ml.streamlit.app/](https://fraud-detection-ml.streamlit.app/)
