# Plan de Estudios: Fundamentos de IA

Nuestro viaje se centrará en el Machine Learning y el Deep Learning, con un enfoque en la comprensión conceptual y la implementación práctica en Python.

## Módulo 1: Fundamentos de Python para IA (Revisión/Preparación)

_(Esto es un recordatorio, ya que hemos usado Python y NumPy. Solo para asegurar que todo esté claro.)_

- **1.1. Repaso Rápido de Python Esencial:**
  - Variables, tipos de datos, estructuras de control (`if/else`, bucles `for`/`while`).
  - Funciones: definición y uso.
  - Listas, Tuplas, Diccionarios y Sets: manipulación básica.
- **1.2. Introducción a NumPy:**
  - Creación y manipulación de arrays (vectores y matrices).
  - Operaciones básicas con arrays (suma, multiplicación elemento a elemento, producto punto).
  - Broadcasting: cómo NumPy maneja operaciones entre arrays de diferentes formas.
- **1.3. Introducción a Matplotlib:**
  - Creación de gráficos simples (dispersión, línea) para visualizar datos y resultados.

**Pregunta de Comprensión:** ¿Te sientes cómodo con estos fundamentos de Python y NumPy? ¿Hay algo que te gustaría repasar brevemente antes de continuar?

---

## Módulo 2: Conceptos Básicos de Machine Learning

- **2.1. ¿Qué es el Machine Learning?**
  - Definición, objetivos y tipos principales.
  - **Aprendizaje Supervisado vs. No Supervisado:** Explicación clara y ejemplos.
  - **Regresión vs. Clasificación:** Diferencias fundamentales y ejemplos.
- **2.2. El Primer Algoritmo: Regresión Lineal (Desde Cero)**
  - **Concepto:** Predecir un valor numérico continuo.
  - **La Ecuación de la Línea:** $y = mx + b$ (para una sola característica) y su generalización $y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$ (múltiples características).
  - **Concepto de "Mejor Ajuste":** ¿Cómo definimos qué línea es la "mejor"?
  - **Función de Costo: Error Cuadrático Medio (MSE):**
    - Explicación de por qué usamos el error al cuadrado.
    - Formulación: $MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$
    - **Objetivo:** Minimizar esta función de costo.
  - **Derivación Conceptual del Descenso de Gradiente (Gradient Descent):**
    - La analogía de bajar una montaña (función de costo) para encontrar el punto más bajo (mínimo).
    - El concepto de "gradiente" como la pendiente de la montaña en un punto dado.
    - Cómo el gradiente nos indica la dirección de mayor ascenso y, por tanto, la dirección opuesta es el mayor descenso.
    - La actualización de los parámetros (pesos $w$ y sesgo $b$) en la dirección opuesta al gradiente.
  - **Implementación en Python Puro de Gradiente Descendente para Regresión Lineal:**
    - Inicialización aleatoria de $w$ y $b$.
    - Cálculo de la predicción $\hat{y}$.
    - Cálculo del MSE.
    - Cálculo de los "gradientes" (parciales de la función de costo con respecto a $w$ y $b$ - de forma conceptual/intuitiva inicialmente).
    - Actualización de $w$ y $b$ usando la tasa de aprendizaje.
    - Bucle de entrenamiento (épocas).
  - **Visualización:** Graficar los datos, la línea de regresión y cómo la línea se ajusta mejor con cada época. Mostrar cómo el error disminuye.

**Pregunta de Comprensión:** ¿Entendiste la Regresión Lineal, la función de costo MSE y la intuición detrás del Descenso de Gradiente?

- **2.3. Regresión Lineal con Scikit-learn:**
  - Introducción a Scikit-learn: Por qué y cuándo usar librerías.
  - Uso de `LinearRegression` de `sklearn.linear_model`.
  - Comparar la complejidad de la implementación manual vs. la librería.
  - Ventajas: eficiencia, robustez, características adicionales.

**Pregunta de Comprensión:** ¿Ves la diferencia entre implementar desde cero y usar una librería?

---

## Módulo 3: Clasificación Básica

- **3.1. Introducción a la Clasificación:**
  - Ejemplos de problemas de clasificación (spam/no spam, gato/perro).
  - Clasificación binaria vs. multiclase.
- **3.2. Regresión Logística (Desde Cero):**
  - **Concepto:** Aunque se llama "regresión", es un algoritmo de **clasificación**.
  - **La Función Sigmoide:**
    - Explicación: $f(z) = \frac{1}{1 + e^{-z}}$.
    - Propósito: Mapear cualquier valor real a una probabilidad entre 0 y 1.
    - Relación con la suma ponderada del Perceptrón ($z$ es nuestra $S+b$).
  - **Concepto de Probabilidad:** Cómo la salida de la sigmoide se interpreta como la probabilidad de pertenecer a una clase.
  - **Función de Costo (Entropía Cruzada Binaria - Binary Cross-Entropy):**
    - Explicación intuitiva: Penaliza más las predicciones erróneas con alta confianza.
    - Formulación: $L(y, \hat{y}) = -(y \log(\hat{y}) + (1-y) \log(1-\hat{y}))$
    - **Objetivo:** Minimizar esta función de costo.
  - **Descenso de Gradiente para Regresión Logística:**
    - Ajuste similar de pesos y sesgo, pero los gradientes se derivan de la función de costo de Entropía Cruzada.
    - Explicación conceptual de cómo los ajustes se realizan.
  - **Implementación en Python Puro de Regresión Logística con Gradiente Descendente:**
    - Clase `LogisticRegression` similar a `Perceptron`, pero con sigmoide y BCE.
  - **Evaluación de un Modelo de Clasificación:**
    - **Precisión (Accuracy):** La métrica más simple.
    - **Matriz de Confusión:** Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos, Falsos Negativos.
    - **Precisión (Precision), Exhaustividad (Recall), Puntuación F1 (F1-score):** Explicación conceptual de su importancia para problemas desbalanceados.

**Pregunta de Comprensión:** ¿Captaste la idea de la Regresión Logística, la función sigmoide y la función de costo Entropía Cruzada?

- **3.3. Regresión Logística con Scikit-learn:**
  - Uso de `LogisticRegression` de `sklearn.linear_model`.
  - Comparación y ventajas.

**Pregunta de Comprensión:** ¿Cómo te sientes con la transición a librerías después de la implementación manual?

---

## Módulo 4: Redes Neuronales Multicapa (MLP) y Transición a Keras

- **4.1. El Perceptrón y sus Limitaciones (Revisión):**
  - Confirmar por qué un solo Perceptrón no puede resolver problemas no linealmente separables (ej. XOR).
- **4.2. Redes Neuronales Multicapa (MLP - Desde Cero):**
  - **Concepto:** Apilamiento de neuronas en capas (entrada, oculta, salida).
  - **Funciones de Activación No Lineales:**
    - ¿Por qué son necesarias? (Para resolver problemas no lineales).
    - ReLU (Rectified Linear Unit), Tanh (Tangente Hiperbólica), Sigmoid (revisión de su uso en capas ocultas).
  - **Propagación Hacia Adelante (Forward Propagation):**
    - Explicación paso a paso de cómo los datos fluyen desde la entrada hasta la salida, calculando las activaciones de cada neurona en cada capa.
    - Implementación paso a paso de la propagación hacia adelante para un MLP pequeño en Python puro (ej. para resolver XOR).
  - **Retropropagación (Backpropagation):**
    - **Explicación INTUITIVA:** Cómo el error de la capa de salida se propaga hacia atrás a través de la red para ajustar los pesos de las capas anteriores.
    - Analogía de cómo los errores se "distribuyen" hacia atrás para culpar a las neuronas y conexiones responsables.
    - Mencionar la "Regla de la Cadena" del cálculo como base matemática, pero sin exigir derivaciones completas a priori.
    - El objetivo es entender la lógica del ajuste de pesos en todas las capas.
  - **Implementación Simplificada en Python Puro de un MLP Básico con Retropropagación:**
    - Para un problema como XOR, mostrando cómo los pesos se ajustan.
    - El papel del optimizador (e.g., Descenso de Gradiente Estocástico - SGD).

**Pregunta de Comprensión:** ¿Comprendes el concepto de capas ocultas, funciones de activación no lineales, propagación hacia adelante y la idea intuitiva de retropropagación?

- **4.3. Introducción a TensorFlow y Keras:**
  - **Por qué necesitamos estas librerías:** Escalabilidad, eficiencia (GPUs), grafos computacionales, optimización avanzada.
  - **Keras como API de alto nivel para TensorFlow:** Facilita la construcción de redes neuronales.
  - **Construcción de un Modelo Secuencial en Keras:**
    - `Sequential()`: la forma más común de apilar capas.
    - `Dense` (capa densamente conectada/fully connected): qué son y cómo se definen.
    - Parámetros: `units` (número de neuronas), `activation` (función de activación), `input_shape`.
  - **Compilación del Modelo (`model.compile`):**
    - `optimizer`: `adam`, `sgd`, etc. (Explicación conceptual de para qué sirven).
    - `loss`: Función de pérdida (ej. `binary_crossentropy`, `mse`).
    - `metrics`: Métricas para evaluar el rendimiento (ej. `accuracy`).
  - **Entrenamiento del Modelo (`model.fit`):**
    - `epochs`: Número de pasadas sobre los datos.
    - `batch_size`: Cuántos ejemplos se procesan antes de una actualización de pesos.
  - **Evaluación del Modelo (`model.evaluate`):**
    - Obtener el rendimiento del modelo en datos de prueba.
  - **Predicción con el Modelo (`model.predict`):**
    - Obtener las salidas del modelo para nuevas entradas.
  - **Implementación de la Misma Red Neuronal Multicapa con Keras/TensorFlow:**
    - Resolver el problema XOR o un problema similar de toy con Keras.

**Pregunta de Comprensión:** ¿Cómo te sientes con la transición de la implementación manual a Keras? ¿Comprendes la función de los diferentes parámetros al construir y entrenar un modelo Keras?

---

## Módulo 5: Conceptos Clave y Buenas Prácticas en Machine Learning

- **5.1. Overfitting (Sobreajuste) y Underfitting (Subajuste):**
  - **Explicación:** Qué son, cómo se ven en los gráficos de entrenamiento/validación.
  - **Identificación:** Curvas de pérdida y precisión.
  - **Técnicas de Regularización:**
    - **L1/L2 (Regularización de pesos):** Intuición de cómo penalizan pesos grandes para evitar la complejidad excesiva del modelo.
    - **Dropout:** Explicación conceptual de cómo "apagar" aleatoriamente neuronas durante el entrenamiento ayuda a la robustez.
  - **Validación Cruzada (Cross-Validation):** Explicación de K-Fold y por qué es importante para una evaluación robusta del modelo.

**Pregunta de Comprensión:** ¿Entiendes los conceptos de overfitting/underfitting y las principales técnicas para mitigarlos?

- **5.2. Datos y Preprocesamiento:**
  - **La importancia de los datos:** "Garbage in, garbage out."
  - **Carga de Datos:** Usando `pandas` para leer CSV, Excel, etc.
  - **Manejo de Valores Nulos/Ausentes:** Estrategias (eliminación, imputación por media/mediana/moda).
  - **Codificación de Variables Categóricas:**
    - One-Hot Encoding: ¿Por qué y cómo se usa? (`pd.get_dummies` o `sklearn.preprocessing.OneHotEncoder`).
    - Label Encoding.
  - **Escalado de Características:**
    - **Normalización (Min-Max Scaling):** Escalar datos a un rango [0, 1].
    - **Estandarización (Z-score Normalization):** Escalar datos a media 0 y desviación estándar 1.
    - ¿Por qué son importantes para algoritmos basados en distancia y Descenso de Gradiente?

**Pregunta de Comprensión:** ¿Comprendes la importancia y las técnicas básicas de preprocesamiento de datos?

---

## Módulo 6: Introducción a Redes Neuronales Especializadas (Conceptual y Keras)

- **6.1. Redes Neuronales Convolucionales (CNNs):**
  - **Caso de Uso:** Imágenes (y otras cuadrículas de datos).
  - **Concepto de Filtros (Kernels) y Convolución:**
    - ¿Cómo detectan patrones como bordes, texturas?
    - Visualización de filtros.
  - **Capas de Pooling (Max Pooling, Average Pooling):**
    - Propósito: Reducir dimensionalidad y hacer el modelo más robusto a pequeñas variaciones.
  - **Arquitectura Básica de una CNN:** `Conv2D` -> `Pooling` -> `Flatten` -> `Dense` (conceptual).
  - **Implementación Simple de una CNN con Keras (ej. para el dataset MNIST de dígitos):**
    - Carga de datos de imagen.
    - Definición de capas `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`.
    - Entrenamiento y evaluación.

**Pregunta de Comprensión:** ¿Captaste la intuición detrás de las CNNs y sus capas principales?

- **6.2. Redes Neuronales Recurrentes (RNNs):**
  - **Caso de Uso:** Datos secuenciales (texto, series de tiempo, audio).
  - **Concepto de "Memoria" y Bucle Recurrente:**
    - La idea de que la salida de un paso de tiempo se alimenta de nuevo como entrada para el siguiente paso.
    - Cómo manejan las dependencias a lo largo del tiempo.
  - **Limitaciones de RNNs Básicas:** Problemas de desvanecimiento/explosión del gradiente.
  - **Concepto de LSTMs (Long Short-Term Memory) y GRUs (Gated Recurrent Units):**
    - Explicación muy conceptual de las "puertas" (gates) que les permiten recordar información a largo plazo y olvidar información irrelevante.
  - **Implementación Simple de una RNN o LSTM con Keras (ej. para un problema de secuencia muy básico o un ejemplo de juguete de texto):**
    - Definición de capas `SimpleRNN`, `LSTM`, `GRU`.

**Pregunta de Comprensión:** ¿Entiendes el concepto de las RNNs para datos secuenciales y por qué surgieron las LSTMs/GRUs?

---

## Módulo 7: Otros Algoritmos de Machine Learning (Introducción conceptual)

- **7.1. Árboles de Decisión y Random Forests:**
  - Intuición de cómo funcionan, ventajas (interpretabilidad) y desventajas.
- **7.2. Máquinas de Vectores de Soporte (SVM):**
  - La idea de encontrar el "hiperplano" óptimo que maximiza el margen entre clases.
- **7.3. K-Means (Aprendizaje No Supervisado):**
  - Algoritmo de clustering (agrupamiento) para encontrar patrones en datos sin etiquetas.

---

Este plan es ambicioso pero totalmente realizable con nuestro enfoque de construcción desde cero y preguntas de comprensión en cada paso. Estoy listo para el siguiente módulo cuando tú lo estés.

¿Qué te parece este plan? ¿Hay algún tema que te gustaría añadir o ajustar?
