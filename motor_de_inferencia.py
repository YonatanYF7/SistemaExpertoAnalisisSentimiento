# motor_de_inferencia.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def balancear_datos(datos):
    df_neg = datos[datos['sentimiento'] == 'negativo']
    df_neu = datos[datos['sentimiento'] == 'neutral']
    df_pos = datos[datos['sentimiento'] == 'positivo']

    df_neu_upsampled = resample(df_neu, replace=True, n_samples=len(df_neg), random_state=123)
    df_pos_upsampled = resample(df_pos, replace=True, n_samples=len(df_neg), random_state=123)

    datos_bal = pd.concat([df_neg, df_neu_upsampled, df_pos_upsampled])
    return datos_bal

def entrenar_modelo(X_tfidf, y):
    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

    # Entrenamiento del modelo
    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    # Evaluación del modelo
    y_pred = modelo.predict(X_test)
    
    print("Precisión:", accuracy_score(y_test, y_pred))
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
    
    return modelo

def predecir_sentimiento(modelo, vectorizador, texto):
    texto_tfidf = vectorizador.transform([texto])
    prediccion = modelo.predict(texto_tfidf)
    return prediccion[0]
