#Importamos librerias
from utils.utilsML import *


#Cargamos dataset después de la limpieza inicial
df = pd.read_csv("C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/data/processed/training_df.csv")


#-------Escalado manual-------

#Verificamos la cantidad de transacciones fraudulentas.
df_fraud = df.loc[df['isfraud'] == 1]

#Seleccionamos la misma cantidad de transacciones no fraudulentas.
df_non_fraud = df.loc[df['isfraud']==0].iloc[:7396,:]

#Los unimos para crear nuestro dataset reducido
df_red= pd.concat([df_fraud, df_non_fraud], ignore_index=True)

#Mezclamos las transacciones para desordenarlas
df_red = shuffle(df_red)
df_red.reset_index(inplace=True, drop=True)


#--------------------------------
#Definimos X e y
X = df_red.drop('isfraud', axis=1)
y = df_red['isfraud']

#Separamos en train y validation
X_train, X_val,y_train, y_val = train_test_split(X,y,random_state=42,test_size=0.1)


#-----Transformaciones----

#Importamos Scaler y PCA
ruta_scaler = 'C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/modelos/scaler.pkl'
ruta_pca = 'C:/Users/lydia/OneDrive/Escritorio/Proyecto final_fraude/src/modelos/pca_5.pkl'

#Escalamos datos
scaler = pickle.load(open(ruta_scaler, 'rb'))
scaled_x_val = scaler.transform(X_val)
scaled_x_train = scaler.transform(X_train)

#PCA
pca = pickle.load(open(ruta_pca, 'rb'))
pca.fit(scaled_x_train)

X_train_pca = pd.DataFrame(pca.transform(scaled_x_train), columns = ['PC1', 'PC2', 'PC3','PC4','PC5'])
X_val_pca = pd.DataFrame(pca.transform(scaled_x_val), columns=['PC1', 'PC2', 'PC3','PC4','PC5'])

#Gridsearch con scoring = precision
param_grid_xgb = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

xgb = xgboost.XGBClassifier()
gridsearch_xgb2 = GridSearchCV(xgb, param_grid_xgb, scoring='f1', n_jobs=-1, cv=5)
gridsearch_xgb2.fit(X_train_pca, y_train)


#------------Probamos el modelo---------------

best_params = gridsearch_xgb2.best_params_
modelo = xgboost.XGBClassifier(**best_params)

#Entrenamos el modelo
modelo.fit(X_train_pca, y_train)

#Predecimos
y_pred = modelo.predict(X_val_pca)


#------------Guardamos el modelo---------------
guardar_modelo(modelo, "new_model")