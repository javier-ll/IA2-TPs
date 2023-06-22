import numpy as np
import matplotlib.pyplot as plt
import time

# Generador basado en ejemplo del curso CS231 de Stanford:
# CS231n Convolutional Neural Networks for Visual Recognition
# (https://cs231n.github.io/neural-networks-case-study/)


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


# GENERADOR ALTERNATIVO DE DATOS
# GENERADOR ALTERNATIVO DE DATOS
def generar_datos_complejos(cantidad_ejemplos, cantidad_clases, FACTOR_ANGULO, AMPLITUD_ALEATORIEDAD):

    # FUNCION --> Z = -3X + Y^5

    # Cantidad de puntos por cada clase, misma cantidad para cada una.
    n = int(cantidad_ejemplos / cantidad_clases)
    # Entradas: 2 columnas (x1 y x2)
    x = np.zeros((cantidad_ejemplos, 2))
    # 1 columna: la clase correspondiente (t -> "target") Salida deseada
    t = np.zeros(cantidad_ejemplos, dtype="uint8")
    yx = np.zeros(cantidad_ejemplos)
    aux = 16

    for i in range(cantidad_ejemplos):
        if i % n == 0:
            aux *= 0.5
        # Valores para las curvas de nivel
        yx[i] = aux
    randomgen = np.random.default_rng()

    for clase in range(cantidad_clases):

        # Dominio en el intervalo (0, 2) particionado n veces
        x1 = np.linspace(0, 2, n)

        # Generamos un rango con los subíndices de cada punto de esta clase. Este rango se va
        # desplazando para cada clase: para la primera clase los índices están en [0, n-1], para
        # la segunda clase están en [n, (2 * n) - 1], etc.
        indices = range(clase * n, (clase + 1) * n)

        aleat = yx[indices] * (AMPLITUD_ALEATORIEDAD/3) * \
            randomgen.standard_normal(size=n)
        x2 = (yx[indices] + (FACTOR_ANGULO/10) * x1) ** 0.2 + aleat
        x[indices] = np.c_[x1, x2]
        t[indices] = clase

    return x, t


def inicializar_pesos(n_entrada, n_capa_2, n_capa_3):
    randomgen = np.random.default_rng()

    w1 = 0.1 * randomgen.standard_normal((n_entrada, n_capa_2))
    b1 = 0.1 * randomgen.standard_normal((1, n_capa_2))

    w2 = 0.1 * randomgen.standard_normal((n_capa_2, n_capa_3))
    b2 = 0.1 * randomgen.standard_normal((1, n_capa_3))

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def ejecutar_adelante(x, pesos):
    # Funcion de entrada (a.k.a. "regla de propagacion") para la primera capa oculta
    z = x.dot(pesos["w1"]) + pesos["b1"]

    # Funcion de activacion ReLU para la capa oculta (h -> "hidden")
    h = np.maximum(0, z)

    # Salida de la red (funcion de activacion lineal). Esto incluye la salida de todas
    # las neuronas y para todos los ejemplos proporcionados
    y = h.dot(pesos["w2"]) + pesos["b2"]

    return {"z": z, "h": h, "y": y}


def clasificar(x, pesos):
    # Corremos la red "hacia adelante"
    resultados_feed_forward = ejecutar_adelante(x, pesos)

    # Buscamos la(s) clase(s) con scores mas altos (en caso de que haya mas de una con
    # el mismo score estas podrian ser varias). Dado que se puede ejecutar en batch (x
    # podria contener varios ejemplos), buscamos los maximos a lo largo del axis=1
    # (es decir, por filas)
    max_scores = np.argmax(resultados_feed_forward["y"], axis=1)
    # Tomamos el primero de los maximos (podria usarse otro criterio, como ser eleccion aleatoria)
    # Nuevamente, dado que max_scores puede contener varios renglones (uno por cada ejemplo),
    # retornamos la primera columna
    # Obtiene la clase con el puntaje máximo para cada ejemplo

    return max_scores

# x: n entradas para cada uno de los m ejemplos(nxm)
# t: salida correcta (target) para cada uno de los m ejemplos (m x 1)
# pesos: pesos (W y b)


def train(x, t, pesos, learning_rate, epochs):
    # Cantidad de filas (i.e. cantidad de ejemplos)
    m = np.size(x, 0)
    lossVal = []

    for i in range(epochs):

        # Ejecucion de la red hacia adelante
        resultados_feed_forward = ejecutar_adelante(x, pesos)
        y = resultados_feed_forward["y"]
        h = resultados_feed_forward["h"]
        z = resultados_feed_forward["z"]

        # LOSS
        # a. Exponencial de todos los scores
        exp_scores = np.exp(y)

        # b. Suma de todos los exponenciales de los scores, fila por fila (ejemplo por ejemplo).
        #    Mantenemos las dimensiones (indicamos a NumPy que mantenga la segunda dimension del
        #    arreglo, aunque sea una sola columna, para permitir el broadcast correcto en operaciones
        #    subsiguientes)
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)

        # c. "Probabilidades": normalizacion de las exponenciales del score de cada clase (dividiendo por
        #    la suma de exponenciales de todos los scores), fila por fila
        p = exp_scores / sum_exp_scores

        # d. Calculo de la funcion de perdida global. Solo se usa la probabilidad de la clase correcta,
        #    que tomamos del array t ("target")
        loss = (1 / m) * np.sum(-np.log(p[range(m), t]))

        # Mostramos solo cada 1000 epochs
        if i % 1000 == 0:
            print("Loss epoch", i, ":", loss)
        lossVal.append(loss)

        # Extraemos los pesos a variables locales
        w1 = pesos["w1"]
        b1 = pesos["b1"]
        w2 = pesos["w2"]
        b2 = pesos["b2"]

        # Ajustamos los pesos: Backpropagation
        # Para todas las salidas, L' = p (la probabilidad)...
        dL_dy = p
        dL_dy[range(m), t] -= 1  # ... excepto para la clase correcta
        dL_dy /= m

        dL_dw2 = h.T.dot(dL_dy)                         # Ajuste para w2
        dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)   # Ajuste para b2

        dL_dh = dL_dy.dot(w2.T)

        dL_dz = dL_dh       # El calculo dL/dz = dL/dh * dh/dz. La funcion "h" es la funcion de activacion de la capa oculta,
        # para la que usamos ReLU. La derivada de la funcion ReLU: 1(z > 0) (0 en otro caso)
        dL_dz[z <= 0] = 0

        dL_dw1 = x.T.dot(dL_dz)                         # Ajuste para w1
        dL_db1 = np.sum(dL_dz, axis=0, keepdims=True)   # Ajuste para b1

        # Aplicamos el ajuste a los pesos
        w1 += -learning_rate * dL_dw1
        b1 += -learning_rate * dL_db1
        w2 += -learning_rate * dL_dw2
        b2 += -learning_rate * dL_db2

        # Actualizamos la estructura de pesos
        # Extraemos los pesos a variables locales
        pesos["w1"] = w1
        pesos["b1"] = b1
        pesos["w2"] = w2
        pesos["b2"] = b2

    return pesos, lossVal


def iniciar(numero_clases, numero_ejemplos, graficar_datos, datos,FACTOR_ANGULO, AMPLITUD_ALEATORIEDAD ,NEURONAS_CAPA_OCULTA, NEURONAS_ENTRADA, LEARNING_RATE, EPOCHS):
    x = []
    t = []
    # Generamos datos
    if datos == 0:
        # datos simples
        x, t = generar_datos_simples(numero_ejemplos, numero_clases,FACTOR_ANGULO, AMPLITUD_ALEATORIEDAD)

    elif datos == 1:
        # datos complejos
        x, t = generar_datos_complejos(numero_ejemplos, numero_clases, FACTOR_ANGULO,AMPLITUD_ALEATORIEDAD)

    # Graficamos los datos si es necesario
    print(graficar_datos)
    input("Datos Generados [Enter]")
    if graficar_datos:
        # Parametro: "c": color (un color distinto para cada clase en t)
        plt.scatter(x[:, 0], x[:, 1], c=t)
        plt.show()

    # Inicializa pesos de la red
    pesos = inicializar_pesos(n_entrada=NEURONAS_ENTRADA,
                              n_capa_2=NEURONAS_CAPA_OCULTA, n_capa_3=numero_clases)

    # Entrena
    start_time = time.time()
    pesos, lossVal = train(x, t, pesos, LEARNING_RATE, EPOCHS)
    end_time = time.time()

    execution_time = end_time - start_time

    return pesos, x, t, lossVal,execution_time


"""
4 - Experimentar con distintos parámetros de configuración del 
generador de datos para generar sets de datos más complejos 
(con clases más solapadas, o con más clases). 
Alternativamente experimentar con otro generador de datos distinto
(desarrollado por usted). Evaluar el comportamiento de la red ante 
estos cambios.
"""
print("Ejercicio 4")
print("----------------------------------------------------------")
print(" ")

# Variable por defecto generador
FACTOR_ANGULO = 0.79
AMPLITUD_ALEATORIEDAD = 0.1
numero_clases = 3
numero_ejemplos = 300

# Variables por algoritmo
numero_capas_ocultas = 100
numero_neuronas_entrada = 2
learning_rate = 1
epoch = 10000

# Generar nuevos datos de entrada
cantidad_puntos = 10
nuevos_datos = np.random.uniform(-1, 1, (cantidad_puntos, 2))

print(">0: Desactivar graficos \n>1: Activar graficos")
graficar_datos = False if int(input()) == 0 else True

print(">0: Datos simples \n>1: Datos complejos")
datos = int(input())


## INICIAR ##
print(" ")

while True:
    print("----- MENÚ -----")
    print("1. Entrenamiento")
    print("2. Prueba")
    print("3. Prueba Parametros")
    print("4. Configuración")
    print("5. Graficas")
    print("6. Salir")

    opcion = input("Seleccione una opción: ")

    if opcion == "1":
       
       
        pesos, x, t, loss_values, tiempo= iniciar(numero_clases, numero_ejemplos, graficar_datos,
                                           datos,FACTOR_ANGULO, AMPLITUD_ALEATORIEDAD, numero_capas_ocultas, numero_neuronas_entrada, learning_rate, epoch)
        
        # Crear gráfico de loss
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss por Epoch')
        plt.show()

        print("#Tiempos#")
        print(tiempo)
   
    elif opcion == "2":

        # Llamar a la función clasificar()
        # Obtener las etiquetas de clase para los nuevos datos
        etiquetas_nuevos_datos = clasificar(nuevos_datos, pesos)
         # Parametro: "c": color (un color distinto para cada clase en t)
        plt.scatter(x[:, 0], x[:, 1], c=t)
        plt.scatter(nuevos_datos[:, 0], nuevos_datos[:, 1],
                    c=etiquetas_nuevos_datos, marker='x')
        plt.show()

    elif opcion == "3":

        # Valores a iterar
        numero_clases_test = [2, 3, 4, 10]
        numero_ejemplos_test = [300, 500, 1000, 3000]
        FACTOR_ANGULO_test = [0.79, 0.86, 0.92,1]
        AMPLITUD_ALEATORIEDAD_test = [0, 0.1, 0.2, 0.5]
        # Lista para almacenar los valores de loss_values de cada iteración
        loss_values_iteraciones = []
        tiempos_ejecucion = []
        colores = ['blue', 'green', 'red', 'orange']  # Colores para cada gráfica
        i = 0

        # ###### ANALISIS CANTIDAD DE CLASES ########
        # print("###### ANALISIS CANTIDAD DE CLASES ########")


        # for dato in numero_clases_test:
            
        #     # Llamar a la función iniciar
        #     print("Numero de clase: ",dato)
        #     pesos, x, t, loss_values,tiempo = iniciar(dato, numero_ejemplos, graficar_datos, datos, FACTOR_ANGULO, AMPLITUD_ALEATORIEDAD, numero_capas_ocultas, numero_neuronas_entrada, learning_rate, epoch)

        
        #     # Almacenar los valores de loss_values en la lista
        #     loss_values_iteraciones.append(loss_values)

        #     #almacenar los tiempo de ejecucion
        #     tiempos_ejecucion.append(tiempo)
            
        # i = 0
        # for loss_values in loss_values_iteraciones:
        #     plt.plot(loss_values, color=colores[i])
        #     i+=1
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Valores de Loss en diferentes iteraciones')
        # plt.legend(numero_clases_test)
        # plt.show()


        # # Crear el gráfico de barras
        # plt.bar(numero_clases_test,tiempos_ejecucion, color=colores)
        # # Agregar etiquetas y título
        # plt.xlabel('numero_clases_test')
        # plt.ylabel('tiempos_ejecucion')
        # plt.title('Tiempos de ejecucion VS numero de clases')
        # # Mostrar el gráfico
        # plt.show()

        # # ###### ANALISIS CANTIDAD DE EJEMPLOS ########
        # # print("###### ANALISIS CANTIDAD DE EJEMPLOS ########")
        # # tiempos_ejecucion = []
        # # loss_values = []

        # # for dato in numero_ejemplos_test:
        # #     # Llamar a la función iniciar
        # #     print("Numero de Ejemplos: ",dato)
        # #     pesos, x, t, loss_values, tiempo = iniciar(numero_clases, dato, graficar_datos, datos, FACTOR_ANGULO, AMPLITUD_ALEATORIEDAD, numero_capas_ocultas, numero_neuronas_entrada, learning_rate, epoch)
            
        # #     # Almacenar los valores de loss_values en la lista
        # #     loss_values_iteraciones.append(loss_values)

        # #     # almacenar los tiempo de ejecucion
        # #     tiempos_ejecucion.append(tiempo)

        # # print("#Tiempos#")
        # # print(tiempos_ejecucion)
            
        # # i=0
        # # for loss_values in loss_values_iteraciones:
        # #     plt.plot(loss_values, color=colores[i])
        # #     i+=1
        # # plt.xlabel('Epoch')
        # # plt.ylabel('Loss')
        # # plt.title('Valores de Loss en diferentes iteraciones')
        # # plt.legend(numero_ejemplos_test)
        # # plt.show()

        # # # Crear un rango de valores para el eje x
        # # x = np.arange(len(numero_ejemplos_test))

        # # # Crear el gráfico de barras
        # # plt.bar(x, tiempos_ejecucion, color=colores)

        # # # Agregar etiquetas y título
        # # plt.xlabel('numero_ejemplos_test')
        # # plt.ylabel('tiempos_ejecucion')
        # # plt.title('Tiempos de ejecucion VS numero de ejemplos')
        # # # Mostrar el gráfico
        # # plt.show()


        ###### ANALISIS FACTOR_ANGULO_test ########
        print("###### ANALISIS FACTOR_ANGULO_test ########")
        tiempos_ejecucion = []
        loss_values = []

        for dato in FACTOR_ANGULO_test:
            # Llamar a la función iniciar
            print("Numero de Ejemplos: ",dato)
            pesos, x, t, loss_values, tiempo = iniciar(numero_clases, numero_ejemplos , graficar_datos, datos, dato, AMPLITUD_ALEATORIEDAD, numero_capas_ocultas, numero_neuronas_entrada, learning_rate, epoch)
            
            # Almacenar los valores de loss_values en la lista
            loss_values_iteraciones.append(loss_values)
            
            # almacenar los tiempo de ejecucion
            tiempos_ejecucion.append(tiempo)

        print("TIMEPOS")
        print(tiempos_ejecucion)
            
        i = 0
        for loss_values in loss_values_iteraciones:
            plt.plot(loss_values, color=colores[i])
            i+=1
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Valores de Loss en diferentes iteraciones')
        plt.legend(FACTOR_ANGULO_test)
        plt.show()

        # Crear un rango de valores para el eje x
        x = np.arange(len(FACTOR_ANGULO_test))

        # Crear el gráfico de barras
        plt.bar(x, tiempos_ejecucion, color=colores)

        # Agregar etiquetas y título
        plt.xlabel('FACTOR_ANGULO_test')
        plt.ylabel('tiempos_ejecucion')
        plt.title('Tiempos de ejecucion VS Factor de angulo')
        # Mostrar el gráfico
        plt.show()

        # ###### ANALISIS AMPLITUD_ALEATORIEDAD_test ########
        # print("###### ANALISIS AMPLITUD_ALEATORIEDAD_test ########")
        # tiempos_ejecucion = []
        # loss_values = []

        # for dato in AMPLITUD_ALEATORIEDAD_test:
        #     # Llamar a la función iniciar
        #     print("Numero de Ejemplos: ",dato)
        #      # def iniciar(numero_clases, numero_ejemplos, graficar_datos, datos,FACTOR_ANGULO, AMPLITUD_ALEATORIEDAD ,NEURONAS_CAPA_OCULTA, NEURONAS_ENTRADA, LEARNING_RATE, EPOCHS):
        #     pesos, x, t, loss_values, tiempo = iniciar(numero_clases, numero_ejemplos , graficar_datos, datos, FACTOR_ANGULO, dato, numero_capas_ocultas, numero_neuronas_entrada, learning_rate, epoch)
            
        #     # Almacenar los valores de loss_values en la lista
        #     loss_values_iteraciones.append(loss_values)
            
        #     # almacenar los tiempo de ejecucion
        #     tiempos_ejecucion.append(tiempo)

        # print("TIMEPOS")
        # print(tiempos_ejecucion)
            
        # i = 0
        # for loss_values in loss_values_iteraciones:
        #     plt.plot(loss_values, color=colores[i])
        #     i+=1
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Valores de Loss en diferentes iteraciones')
        # plt.legend(AMPLITUD_ALEATORIEDAD_test)
        # plt.show()

        # # Crear un rango de valores para el eje x
        # x = np.arange(len(AMPLITUD_ALEATORIEDAD_test))

        # # Crear el gráfico de barras
        # plt.bar(x, tiempos_ejecucion, color=colores)

        # # Agregar etiquetas y título
        # plt.xlabel('FACTOR_ANGULO_test')
        # plt.ylabel('tiempos_ejecucion')
        # plt.title('Tiempos de ejecucion VS Factor de angulo')
        # # Mostrar el gráfico
        # plt.show()

        

    elif opcion == "4":

        print("----- MENÚ Conf.-----")
        print("1.Configuracion Generador de datos")
        print("2.Configuracion Algoritmo")
        opcion = int(input())

        if opcion == 1:

            print("Numero de Clases (defecto 3)")
            numero_clases = int(input())
            print("Numero de Ejemplos (defecto 300)")
            numero_ejemplos = int(input())

            print("FACTOR ANGULO (defecto 0.79)")
            FACTOR_ANGULO = float(input())
            print("AMPLITUD LEATORIEDAD (defecto 0.1)")
            AMPLITUD_ALEATORIEDAD = float(input())

        elif opcion == 2:

            print("Numero de Capas ocultas (defecto 100)")
            numero_capas_ocultas = int(input())
            print("Numero de neuronas de entrada (defecto 2)")
            numero_neuronas_entrada = int(input())
            print("Learning Rate (defecto 1)")
            learning_rate = int(input())
            print("Cantidad de Epoch  (defecto 10000)")
            epoch = int(input())
        
        opcion = 1

    elif opcion == "5":

        # Parametro: "c": color (un color distinto para cada clase en t)
        plt.scatter(x[:, 0], x[:, 1], c=t)
        plt.scatter(nuevos_datos[:, 0], nuevos_datos[:, 1],
                    c=etiquetas_nuevos_datos, marker='x')
        plt.show()

        # Crear gráfico de loss
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss por Epoch')
        plt.show()

    elif opcion == "6":
        break
    else:
        print("Opción inválida. Intente nuevamente.")
