# ==============================================================================
# ======================== LIBRERÍAS ===========================================
# ==============================================================================
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn.model_selection import train_test_split
#Preprocesado
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
#Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
#Métricas, 
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, accuracy_score,precision_score
#Guardar modelos
import pickle

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')



# ==============================================================================
# ============================FUNCIONES ========================================
# ==============================================================================

#Función para graficar la densidad de  todas las columnas numéricas con un gráfico lineal
def dens_uni(df):
    '''
    Función que grafica la densidad de todas las columnas numéricas con un gráfico lineal. 
    '''
    for col in df.columns:
        sns.kdeplot(df[col], shade=True)
        sns.color_palette("husl",5)
        plt.xlabel(col)
        plt.show()
    return df


#Función para guardar csv en train y test
def dataset_to_train_test(df: pd.DataFrame, test_size: float, random_state: int, path_train: str, path_test: str):

    """
    Objectivo:
    ---
    Divide un dataset en conjuntos de entrenamiento y prueba, y los guarda en archivos CSV.

    args.
    ---
    df: DataFrame; el DataFrame que se desea dividir y guardar.

    test_size: float; el tamaño del conjunto de prueba. Debe estar entre 0 y 1.

    random_state: int; semilla aleatoria para reproducibilidad de los resultados.

    ruta_train: str; ruta donde se guardará el archivo CSV del conjunto de entrenamiento.

    ruta_test: str; ruta donde se guardará el archivo CSV del conjunto de prueba.
    """

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Guardar los conjuntos de entrenamiento y prueba en archivos CSV
    train_df.to_csv(path_train, index=False)
    test_df.to_csv(path_test, index=False)

    print("Los conjuntos de entrenamiento y prueba se han guardado exitosamente.")

def comparar_metricas_modelos(model_names, X_test, y_test):
  '''
  Función que devuelve una gráfica lineal que muestra las métricas de nuestros modelos de clasificación.
  ---
  Imprescindible generar previamente una lista con los nombres de los modelos tal y como están guardados en pickle.
  '''
  # Listas vacías para almacenar los valores de las métricas y los nombres de los modelos
  f1_scores = []
  accuracy_scores = []
  recall_scores = []
  precision_scores = []
  sorted_model_names = []

  # Cargamos los modelos desde los archivos pickle y calculamos las métricas
  for model_name in model_names:
      file_path = f'/content/drive/MyDrive/Colab Notebooks/Proyecto final_fraude/src/modelos/{model_name}.pkl'
      with open(file_path, 'rb') as f:
          model = pickle.load(f)
      y_pred = model.predict(X_val_pca)

      # Calculamos las métricas y las almacenamos en las listas correspondientes
      f1 = f1_score(y_val, y_pred)
      f1_scores.append(f1)

      accuracy = accuracy_score(y_val, y_pred)
      accuracy_scores.append(accuracy)

      recall = recall_score(y_val, y_pred)
      recall_scores.append(recall)

      precision = precision_score(y_val, y_pred)
      precision_scores.append(precision)

      sorted_model_names.append(model_name)

  # Ordenamos los F1-scores
  f1_scores, sorted_model_names = zip(*sorted(zip(f1_scores, sorted_model_names), reverse=True))

  # Datos estéticos de la gráfica
  line_styles = ['-'] * len(sorted_model_names)
  plt.figure(figsize=(10, 6))
  plt.xticks(range(len(sorted_model_names)), sorted_model_names, rotation=90)

  # Creamos la gráfica
  plt.plot(f1_scores, label='F1-score', color='blue', linestyle='-', marker='o')
  plt.plot(accuracy_scores, label='Accuracy', color='green', linestyle='-', marker='o')
  plt.plot(recall_scores, label='Recall', color='orange', linestyle='-', marker='o')
  plt.plot(precision_scores, label='Precision', color='red', linestyle='-', marker='o')

  # Títulos y leyendas
  plt.title('Comparación de métricas de los modelos')
  plt.xlabel('Modelos')
  plt.ylabel('Métricas')
  plt.legend()
  plt.show()


#Función para probar varios modelos con datos en crudo
def probar_modelos_clasificacion(X_train, y_train, X_test, y_test,nombre_modelo="modelo_xx"):
  '''
  Función que prueba varios modelos de clasificación con nuestros datos, entrenarlos y predecir resultados
  ---
  Modelos por defecto: 
    - Regresión Logística
    - Árbol de decisión
    - KNN
    - Random Forest
    - XGBoost
  Los modelos tienen los hiperparámetros que vienen por defecto.
  ---
  Métricas porq defecto:

    - Matriz de confusión
    - Informe de clasificación (muestra accuracy, recall, precision y f1-score)
  '''
    # Crear una lista de modelos
  modelos = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        RandomForestClassifier(),
        xgboost.XGBClassifier(random_state=42)
    ]

  # Realizar pruebas para cada modelo
  for modelo in modelos:
      # Entrenar el modelo
      modelo.fit(X_train, y_train)

      # Realizar predicciones en el conjunto de prueba
      y_pred = modelo.predict(X_test)

      #Guardar modelo
      guardar_modelo(modelo, f"{type(modelo).__name__}_{nombre_modelo}")


      # Calcular y mostrar las métricas de evaluación
      print(f"Modelo: {type(modelo).__name__}")
      print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
      print(classification_report(y_test, y_pred))
      print("------------------------------------")
  return y_pred,modelo



def guardar_modelo(modelo, nombre_modelo):
  '''
  Función que toma como entrada un modelo de machine learning y un nombre para el modelo 
  y se encarga de guardar el modelo en un archivo utilizando pickle.dump().
  El modelo se guarda en la dirección añadida manualmente en "directorio modelos".
  '''
  directorio_modelos = "/content/drive/MyDrive/Colab Notebooks/Proyecto final_fraude/src/modelos"

  if not os.path.exists(directorio_modelos):
        os.makedirs(directorio_modelos)

  ruta_modelo = os.path.join(directorio_modelos, nombre_modelo + ".pkl")
  with open(ruta_modelo, "wb") as file:
      pickle.dump(modelo, file)
  print(f"Modelo {nombre_modelo} guardado exitosamente.")

def  probar_modelos_grid(X_train, y_train, X_test, y_test,nombre_modelo="modelo_xx"):
  '''
  Función que prueba varios modelos de clasificación con nuestros datos, entrenarlos y predecir resultados
  ---
  Modelos por defecto: 
    - Regresión Logística
    - Árbol de decisión
    - KNN
    - Random Forest
    - XGBoost
  Los modelos tienen los hiperparámetros definidos en el último grid search.
  ---
  Métricas porq defecto:

    - Matriz de confusión
    - Informe de clasificación (muestra accuracy, recall, precision y f1-score)
  '''
    # Mejores parámetros de Logistic Regression
  best_params_lr = grid_params['lr']
  best_params_tree = grid_params['tree'] 
  best_params_knn = grid_params['knn'] 
  best_params_rf = grid_params['rf'] 
  best_params_xgb = grid_params['xgb']

    # Crear una lista de modelos
  modelos = [
      LogisticRegression(**best_params_lr),
      DecisionTreeClassifier(**best_params_tree),
      KNeighborsClassifier(**best_params_knn),
      RandomForestClassifier(**best_params_rf),
      xgboost.XGBClassifier(**best_params_xgb)
    ]

  # Realizar pruebas para cada modelo
  for modelo in modelos:
      # Entrenar el modelo
      modelo.fit(X_train, y_train)

      # Realizar predicciones en el conjunto de prueba
      y_pred = modelo.predict(X_test)

      #Guardar modelo
      guardar_modelo(modelo, f"{type(modelo).__name__}_{nombre_modelo}")


      # Calcular y mostrar las métricas de evaluación
      print(f"Modelo: {type(modelo).__name__}")
      print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
      print(classification_report(y_test, y_pred))
      print("------------------------------------")
  return y_pred,modelo

def probar_modelos_save(X_train, y_train, X_test, y_test):
  '''
  Función que prueba varios modelos de clasificación con nuestros datos.Posteriormente entrena y predice resultados. 
  ---
  Modelos por defecto: 
    - Regresión Logística
    - Árbol de decisión
    - KNN
    - Random Forest
    - XGBoost
  Los modelos tienen los hiperparámetros que vienen por defecto.
  ---
  Métricas porq defecto:

    - Matriz de confusión
    - Informe de clasificación (muestra accuracy, recall, precision y f1-score)
  '''
    # Crear una lista de modelos
  modelos = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        RandomForestClassifier(),
        xgboost.XGBClassifier(random_state=42)
    ]

  for modelo in modelos:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Calcular métricas de evaluación
    f1 = f1_score(y_test, y_pred)

    # Verificar si las métricas cumplen con un umbral deseado
    if f1 > 0.8:
        guardar_modelo(modelo, type(modelo).__name__)

    # Imprimir métricas de evaluación
    print(f"Modelo: {type(modelo).__name__}_v1")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("------------------------------------")

  return y_pred, modelo

def modelos_hyperparam_save(X_train, y_train, X_test, y_test, grid_params):
  '''
  Función que prueba varios modelos de clasificación con los hiperparámetros sacados de un GridSearch previo. Posteriormente entrena y predice resultados. 
  ---
  Modelos por defecto:
    - Regresión Logística
    - Árbol de decisión
    - KNN
    - Random Forest
    - XGBoost
  ---
  Métricas porq defecto:

    - Matriz de confusión
    - Informe de clasificación (muestra accuracy, recall, precision y f1-score)
  ---
  Dentro de esta función,se llama a la función guardar_modelo
  cuando el f1-score del modelo cumpla con los umbrales deseados.
  '''
  
  #Mejores parámetros de cada modelo
  best_params_lr = grid_params['lr'] 
  best_params_tree = grid_params['tree'] 
  best_params_knn = grid_params['knn'] 
  best_params_rf = grid_params['rf'] 
  best_params_xgb = grid_params['xgb'] 

  # Crear una lista de modelos con los mejores parámetros
  modelos = [
      LogisticRegression(**best_params_lr),
      DecisionTreeClassifier(**best_params_tree),
      KNeighborsClassifier(**best_params_knn),
      RandomForestClassifier(**best_params_rf),
      xgboost.XGBClassifier(**best_params_xgb)
    ]

  for modelo in modelos:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Calcular métricas de evaluación
    f1 = f1_score(y_test, y_pred)

    # Verificar si las métricas cumplen con un umbral deseado
    if f1 > 0.80:
        guardar_modelo(modelo, type(modelo).__name__)

    # Imprimir métricas de evaluación
    print(f"Modelo: {type(modelo).__name__}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("------------------------------------")

  return y_pred, modelo

def cargar_modelo(nombre_modelo):
  '''
  Función que permite cargar los modelos guardados en la función "guardar_modelo"
  '''
  directorio_modelos = "modelos"  # Ruta del directorio "modelos"
  ruta_modelo = os.path.join(directorio_modelos, nombre_modelo + ".pkl")

  with open(ruta_modelo, "rb") as file:
      modelo = pickle.load(file)
  return modelo