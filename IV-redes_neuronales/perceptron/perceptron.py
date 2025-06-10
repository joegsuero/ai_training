import numpy as np


class Perceptron:
    def __init__(self, num_entradas, umbral=0):
        self.pesos = np.zeros(num_entradas)
        self.sesgo = 0
        self.umbral = umbral

    def predecir(self, entradas):
        suma_ponderada = np.dot(entradas, self.pesos) + self.sesgo
        if suma_ponderada >= self.umbral:
            return 1
        else:
            return 0

    def entrenar(self, datos_entrenamiento, etiquetas, tasa_aprendizaje=0.1, n_epocas=100):
        """
        Entrena el Perceptrón usando la regla de aprendizaje del Perceptrón.

        Args:
            datos_entrenamiento (np.array): Matriz de entradas (ej: [[0,0], [0,1], ...]).
            etiquetas (np.array): Vector de salidas deseadas (ej: [0,1,0,1]).
            tasa_aprendizaje (float): El tamaño del paso para ajustar los pesos y el sesgo.
            n_epocas (int): El número de veces que se recorrerá todo el conjunto de entrenamiento.
        """
        for epoca in range(n_epocas):
            errores_en_epoca = 0  # Para contar cuántos errores hubo en esta época

            # Recorrer cada ejemplo de entrenamiento
            for entradas_ejemplo, salida_deseada in zip(datos_entrenamiento, etiquetas):
                prediccion = self.predecir(entradas_ejemplo)
                error = salida_deseada - prediccion

                if error != 0:  # Solo ajustamos si hay un error
                    errores_en_epoca += 1
                    # Ajustar pesos
                    # Cada peso se ajusta en proporción al error, la tasa de aprendizaje y la entrada correspondiente.
                    self.pesos += tasa_aprendizaje * error * entradas_ejemplo
                    # Ajustar sesgo
                    self.sesgo += tasa_aprendizaje * error

                    # Si la entrada es cero y hay error, hara cero la suma que hay que hacerle al peso,
                    # por ende, afectara solo al sesgo

            # Opcional: Imprimir el progreso
            print(f"Época {epoca+1}/{n_epocas}, Errores: {errores_en_epoca}")

            # Si no hubo errores en esta época, el Perceptrón ha aprendido y podemos detenernos.
            if errores_en_epoca == 0:
                print(
                    f"Entrenamiento completado en la época {epoca+1}. No más errores.")
                break

        print(
            f"\nEntrenamiento finalizado. Pesos finales: {self.pesos}, Sesgo final: {self.sesgo}")
