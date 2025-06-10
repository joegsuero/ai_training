from perceptron import Perceptron
import numpy as np
# Datos de entrenamiento
# Cada fila es una entrada [¿Llueve?, ¿Hace frío?]
datos_entrenamiento = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Etiquetas (salida deseada) para cada entrada
# [Salir si no llueve y no hace frío, No salir si hace frío, No salir si llueve, No salir si llueve y hace frío]
etiquetas = np.array([1, 0, 0, 0])

# Crear y entrenar el Perceptrón
mi_perceptron_entrenado = Perceptron(num_entradas=2)
mi_perceptron_entrenado.entrenar(
    datos_entrenamiento, etiquetas, tasa_aprendizaje=0.1, n_epocas=100)

# Probar el Perceptrón después del entrenamiento
print("\n--- Pruebas después del entrenamiento ---")
for i, (entradas, etiqueta_verdadera) in enumerate(zip(datos_entrenamiento, etiquetas)):
    prediccion = mi_perceptron_entrenado.predecir(entradas)
    print(
        f"Entrada: {entradas} (¿Llueve?, ¿Frío?) -> Predicción: {prediccion}, Real: {etiqueta_verdadera}")

# Probar con un nuevo escenario
print("\n--- Nuevo escenario ---")
# Si no llueve y no hace frío, debería predecir 1
pred_nueva = mi_perceptron_entrenado.predecir(np.array([0, 0]))
# Esperado: 1
print(f"Predicción para [No llueve, No hace frío]: {pred_nueva}")

# Si llueve y no hace frío, debería predecir 0
pred_nueva_2 = mi_perceptron_entrenado.predecir(np.array([1, 0]))
# Esperado: 0
print(f"Predicción para [Sí llueve, No hace frío]: {pred_nueva_2}")
