# base_de_conocimiento.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def cargar_datos():
    # Cargar el archivo CSV
    datos = pd.read_csv('datos_sentimientos.csv')
    return datos

def preprocesar_datos(datos):
    X = datos['texto']
    y = datos['sentimiento']
    
    # Vectorización TF-IDF
    vectorizador = TfidfVectorizer()
    X_tfidf = vectorizador.fit_transform(X)
    
    return X_tfidf, y, vectorizador

