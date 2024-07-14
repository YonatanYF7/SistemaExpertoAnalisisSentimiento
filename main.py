# main.py

from base_de_conocimiento import cargar_datos, preprocesar_datos
from motor_de_inferencia import balancear_datos, entrenar_modelo, predecir_sentimiento
from interfaz_de_usuario import mostrar_menu, obtener_texto_usuario, mostrar_resultado

def main():
    # Cargar y preprocesar datos
    datos = cargar_datos()
    datos_bal = balancear_datos(datos)
    X_tfidf, y, vectorizador = preprocesar_datos(datos_bal)

    # Entrenar modelo
    modelo = entrenar_modelo(X_tfidf, y)

    # Interfaz de usuario
    while True:
        opcion = mostrar_menu()
        if opcion == '1':
            texto = obtener_texto_usuario()
            prediccion = predecir_sentimiento(modelo, vectorizador, texto)
            mostrar_resultado(prediccion)
        elif opcion == '2':
            break
        else:
            print("Opción no válida. Por favor, selecciona de nuevo.")

if __name__ == "__main__":
    main()
