import matplotlib.pyplot as plt
import random
import math

def temple_simulado(costosFinal, i):
    # Creo una lista n con los productos de la orden
    n = []
    for costo in costosFinal:
        if costo[0]=='i':
            n.append(costo[1])
    # Inicializar camino actual con los productos en orden
    camino_actual = list(n)
    random.shuffle(camino_actual)
    camino_actual.insert(0, i)

    # Inicializar temperatura y factor de enfriamiento
    temperatura = 100.0
    factor_enfriamiento = 0.99
    energias=[]      # Para graficar la evolución de la energía, es decir, la variación del costo 
                     #a lo largo de las iteraciones del algoritmo de Temple Simulado
    # Ciclo principal
    while temperatura > 1.0:
        # Seleccionar vecino aleatorio
        vecino = camino_actual
        j = random.randint(0, len(n)-1)
        k = random.randint(0, len(n)-1)
        while j == k or vecino[j] == 0 or vecino[k] == 0:
            j = random.randint(0, len(n)-1)
            k = random.randint(0, len(n)-1)
        vecino[j], vecino[k] = vecino[k], vecino[j]
        
        camino_actual = [x for x in camino_actual if x != i]
        camino_actual.insert(0, 'i')
        vecino = [x for x in vecino if x != i]
        vecino.insert(0, 'i')


        # Calcular diferencia de costo entre camino actual y vecino
        costo_actual=calcular_costo(camino_actual,costosFinal)
        costo_vecino=calcular_costo(vecino,costosFinal)
        diferencia_costo = costo_vecino - costo_actual
        
        # Si el vecino es mejor, aceptarlo como nuevo camino
        if diferencia_costo < 0:
            camino_actual = vecino
        # Si el vecino es peor, aceptarlo con una cierta probabilidad
        else:
            probabilidad = math.exp(-diferencia_costo / temperatura)
            if random.random() < probabilidad:
                camino_actual = vecino
        
        # Reducir la temperatura
        temperatura *= factor_enfriamiento
        energias.append(costo_actual)
    return camino_actual, energias

#Función para calcular los costos parciales y el de mejor camino
def calcular_costo(camino_actual,costosFinal):
    costo_actual=0
    for i in range(len(camino_actual)-1):
        if [camino_actual[i], camino_actual[i+1]] in [elem[:2] for elem in costosFinal]:
            j = [elem[:2] for elem in costosFinal].index([camino_actual[i], camino_actual[i+1]])
        if [camino_actual[i+1], camino_actual[i]] in [elem[:2] for elem in costosFinal]:
            j = [elem[:2][::-1] for elem in costosFinal].index([camino_actual[i], camino_actual[i+1]])
        costo_actual += costosFinal[j][2]
    return costo_actual

def mejor_camino(costosFinal, inicio, n):
    path = [inicio]
    for i in range(n):
        path, energias = temple_simulado(costosFinal,inicio)
    
    graficar_energia(energias)
    costo_final=calcular_costo(path,costosFinal)
    print("Costo total:", costo_final)
    return path


def graficar_enfriamiento(T0, T_min, alpha):
    T = T0
    temps = [T]
    while T > T_min:
        T *= alpha
        temps.append(T)
    plt.plot(temps)
    plt.ylabel('Temperatura')
    plt.xlabel('Iteración')
    plt.show()


def graficar_energia(energias):
    # Graficamos la evolución de la energía
    plt.plot(energias)
    plt.ylabel('Energía')
    plt.xlabel('Iteración')
    plt.show()


def iniciar_temple(costosFinal):
    inicio = costosFinal[0][0]
    n = 10 # número de iteraciones de Temple Simulado que se realizarán para buscar un mejor camino
    mejor = mejor_camino(costosFinal, inicio, n)
    print("Mejor camino:", mejor)
    T0 = 1.0
    T_min = 0.00001
    alpha = 0.9
    graficar_enfriamiento(T0, T_min, alpha)
    return costosFinal, inicio


#------------------------------------------ TEST ------------------------------------------#
if __name__ == "__main__":
    #"""
    costos = [['i', '10', 8], ['i', '17', 11], ['i', '22', 8], ['i', '25', 3], ['10', '17', 8], ['10', '22', 5], ['10', '25', 7], ['17', '22', 4], ['17', '25', 12], ['22', '25', 9]]
    #costos = [('i', 0, 0), ('a', 1, 1), ('b', 2, 2), ('c', 3, 3), ('d', 4, 4), ('e', 5, 5), ('f', 6, 6), ('g', 7, 7), ('h', 8, 8), ('i', 9, 9)]
    print("costos:\n", costos)
    costosFinal, inicio = iniciar_temple(costos)
    print("costos final: ", costosFinal)
    print("inicio: ", inicio)
    #"""