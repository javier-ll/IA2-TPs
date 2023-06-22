import numpy as np
import matplotlib.pyplot as plt
import time


def generar_datos_simples(cantidad_ejemplos, cantidad_clases, FACTOR_ANGULO, AMPLITUD_ALEATORIEDAD):

    # Calculamos la cantidad de puntos por cada clase, asumiendo la misma cantidad para cada
    # una (clases balanceadas)
    n = int(cantidad_ejemplos / cantidad_clases)

    # Entradas: 2 columnas (x1 y x2)
    x = np.zeros((cantidad_ejemplos, 2))
    # Salida deseada ("target"): 1 columna que contendra la clase correspondiente (codificada como un entero)
    # 1 columna: la clase correspondiente (t -> "target")
    t = np.zeros(cantidad_ejemplos, dtype="uint8")

    randomgen = np.random.default_rng()

    # Por cada clase (que va de 0 a cantidad_clases)...
    for clase in range(cantidad_clases):
        # Tomando la ecuacion parametrica del circulo (x = r * cos(t), y = r * sin(t)), generamos
        # radios distribuidos uniformemente entre 0 y 1 para la clase actual, y agregamos un poco de
        # aleatoriedad
        radios = np.linspace(0, 1, n) + AMPLITUD_ALEATORIEDAD * \
            randomgen.standard_normal(size=n)

        # ... y angulos distribuidos tambien uniformemente, con un desfasaje por cada clase
        angulos = np.linspace(clase * np.pi * FACTOR_ANGULO,
                              (clase + 1) * np.pi * FACTOR_ANGULO, n)

        # Generamos un rango con los subindices de cada punto de esta clase. Este rango se va
        # desplazando para cada clase: para la primera clase los indices estan en [0, n-1], para
        # la segunda clase estan en [n, (2 * n) - 1], etc.
        indices = range(clase * n, (clase + 1) * n)

        # Generamos las "entradas", los valores de las variables independientes. Las variables:
        # radios, angulos e indices tienen n elementos cada una, por lo que le estamos agregando
        # tambien n elementos a la variable x (que incorpora ambas entradas, x1 y x2)
        x1 = radios * np.sin(angulos)
        x2 = radios * np.cos(angulos)
        x[indices] = np.c_[x1, x2]

        # Guardamos el valor de la clase que le vamos a asociar a las entradas x1 y x2 que acabamos
        # de generar
        t[indices] = clase

    return x, t

# Valores a iterar
numero_clases_test = [2, 3, 4, 10]
numero_ejemplos_test = [300, 500, 1000, 3000]
FACTOR_ANGULO_test = [0.79]
AMPLITUD_ALEATORIEDAD_test = [0, 0.1, 0.2, 0.5]
# Lista para almacenar los valores de loss_values de cada iteración
loss_values_iteraciones = []
tiempos_ejecucion = []
colores = ['blue', 'green', 'red', 'orange']  # Colores para cada gráfica
i = 0

for dato in FACTOR_ANGULO_test:
    x, t = generar_datos_simples(3,300,dato, 0.1)

input("Datos Generados [Enter]")

# Parametro: "c": color (un color distinto para cada clase en t)
plt.scatter(x[:, 0], x[:, 1], c=t)
plt.show()

