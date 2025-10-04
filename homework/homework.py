# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable 'default payment next month' corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta 'files/input/'.
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna 'default payment next month' a 'default'.
# - Remueva la columna 'ID'.
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría 'others'.
# - Renombre la columna 'default payment next month' a 'default'
# - Remueva la columna 'ID'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como 'files/models/model.pkl.gz'.
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {'predicted_0': 15562, 'predicte_1': 666}, 'true_1': {'predicted_0': 3333, 'predicted_1': 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {'predicted_0': 15562, 'predicte_1': 650}, 'true_1': {'predicted_0': 2490, 'predicted_1': 1420}}
#
import os
import json
import gzip
import pickle
from glob import glob

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def cargar_datos(ruta_train, ruta_test):
    # Carga de los datos desde archivos comprimidos
    df_train = pd.read_csv(ruta_train, compression='zip')
    df_test = pd.read_csv(ruta_test, compression='zip')
    
    for df in [df_train, df_test]:
        df.rename(columns={'default payment next month': 'default'}, inplace=True)
        if 'ID' in df.columns:
            df.drop(columns=['ID'], inplace=True)
    return df_train, df_test


def limpiar_datos(df):
    # Se filtran registros con valores inválidos en EDUCATION y MARRIAGE
    df_filtrado = df[(df['MARRIAGE'] != 0) & (df['EDUCATION'] != 0)].copy()
    # Para EDUCATION, agrupar valores mayores o iguales a 4 en la categoría 'others' (4)
    df_filtrado['EDUCATION'] = df_filtrado['EDUCATION'].apply(lambda x: 4 if x >= 4 else x)
    return df_filtrado.dropna()


def crear_pipeline_modelo(vars_cat, vars_num):
    # Se define el preprocesador: one-hot encoding para categóricas y escalado estándar para numéricas
    preprocesador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), vars_cat),
            ('num', StandardScaler(), vars_num)
        ]
    )
    
    # Se crea el pipeline con transformación, selección, reducción y clasificación
    pipeline = Pipeline([
        ('preproceso', preprocesador),
        ('selector', SelectKBest(score_func=f_classif)),
        ('reductor', PCA()),
        ('clasificador', MLPClassifier(max_iter=15000, random_state=21))
    ])
    return pipeline


def calcular_metricas(y_true, y_pred, etiqueta):
    return {
        'type': 'metrics',
        'dataset': etiqueta,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }


def obtener_confusion(y_true, y_pred, etiqueta):
    cm = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': etiqueta,
        'true_0': {'predicted_0': int(cm[0, 0]), 'predicted_1': int(cm[0, 1])},
        'true_1': {'predicted_0': int(cm[1, 0]), 'predicted_1': int(cm[1, 1])}
    }


def main():
    ruta_train = './files/input/train_data.csv.zip'
    ruta_test = './files/input/test_data.csv.zip'
    
    # Cargar y limpiar los datos
    df_train, df_test = cargar_datos(ruta_train, ruta_test)
    df_train = limpiar_datos(df_train)
    df_test = limpiar_datos(df_test)
    
    # Separar variables predictoras y objetivo
    X_train = df_train.drop(columns=['default'])
    y_train = df_train['default']
    X_test = df_test.drop(columns=['default'])
    y_test = df_test['default']
    
    # Definir variables categóricas y numéricas
    vars_categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']
    vars_numericas = [col for col in X_train.columns if col not in vars_categoricas]
    
    # Crear el pipeline del modelo
    pipeline = crear_pipeline_modelo(vars_categoricas, vars_numericas)
    
    # Definir la grilla de hiperparámetros
    parametros = {
        'selector__k': [20],
        'reductor__n_components': [None],
        'clasificador__hidden_layer_sizes': [(50, 30, 40, 60)],
        'clasificador__alpha': [0.26],
        'clasificador__learning_rate_init': [0.001]
    }
    
    # Configurar y ajustar GridSearchCV
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        refit=True
    )
    busqueda.fit(X_train, y_train)
    
    # Guardar el modelo en formato comprimido
    ruta_modelos = 'files/models'
    if os.path.exists(ruta_modelos):
        for archivo in glob(os.path.join(ruta_modelos, '*')):
            os.remove(archivo)
        os.rmdir(ruta_modelos)
    os.makedirs(ruta_modelos, exist_ok=True)
    with gzip.open(os.path.join(ruta_modelos, 'model.pkl.gz'), 'wb') as file_modelo:
        pickle.dump(busqueda, file_modelo)
    
    # Realizar predicciones
    pred_train = busqueda.predict(X_train)
    pred_test = busqueda.predict(X_test)
    
    # Calcular métricas y matrices de confusión
    metricas_train = calcular_metricas(y_train, pred_train, 'train')
    metricas_test = calcular_metricas(y_test, pred_test, 'test')
    cm_train = obtener_confusion(y_train, pred_train, 'train')
    cm_test = obtener_confusion(y_test, pred_test, 'test')
    
    # Guardar resultados en un archivo JSON (cada línea un diccionario)
    os.makedirs('files/output', exist_ok=True)
    ruta_metricas = os.path.join('files/output', 'metrics.json')
    with open(ruta_metricas, 'w', encoding='utf-8') as archivo:
        for registro in [metricas_train, metricas_test, cm_train, cm_test]:
            archivo.write(json.dumps(registro) + '\n')


if __name__ == '__main__':
    main()