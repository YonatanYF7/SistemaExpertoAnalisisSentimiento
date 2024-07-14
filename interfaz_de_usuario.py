# interfaz_de_usuario.py

def mostrar_menu():
    print("Sistema de Análisis de Sentimientos")
    print("1. Analizar un nuevo texto")
    print("2. Salir")
    opcion = input("Selecciona una opción: ")
    return opcion

def obtener_texto_usuario():
    texto = input("Introduce el texto a analizar: ")
    return texto

def mostrar_resultado(prediccion):
    print(f"El sentimiento del texto es: {prediccion}")

