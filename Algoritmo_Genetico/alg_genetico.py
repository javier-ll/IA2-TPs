import numpy as np
import random
import time
from layout4 import Layout
import copy
import os

#from ordenes import*
from lector_ordenes import*
from layout_grafico import*

class Individuo:
    def __init__(self, layout, lista_productos, ordenes_picking):
        self.layout = layout
        self.lista_productos = lista_productos
        self.cargar_productos()
        self.ordenes_picking = ordenes_picking
        self.fitness = self.calcular_fitness()
        self.orden_estantes = self.ordenar_por_fila()

    def ordenar_por_fila(self):
        productos = []
        for fila in self.layout:
            productos += [elem for elem in fila if elem != 0 and elem != -2]
        return productos

    def cargar_productos(self):
        numeros_productos = list(range(1, len(self.lista_productos)+1))
        for i in range(len(self.layout)):
            for j in range(len(self.layout[0])):
                if self.layout[i][j] != 0 and self.layout[i][j] != -2:
                    producto = random.choice(numeros_productos)
                    self.layout[i][j] = producto
                    numeros_productos.remove(producto)

    def calcular_fitness(self):
        # Calcula el costo promedio de todas las órdenes en el layout actual
        costos_ordenes = [calcular_costo(self.layout, orden) for orden in self.ordenes_picking]
        fitness = sum(costos_ordenes) / len(costos_ordenes)
        return fitness



class Poblacion:
    def __init__(self, tamano_poblacion, lista_productos, layout_inicial, ordenes, metodo_seleccion):
        self.tamano_poblacion = tamano_poblacion
        self.lista_productos = lista_productos
        self.ordenes = ordenes
        self.individuos = [Individuo(copy.deepcopy(layout_inicial), self.lista_productos, self.ordenes) for _ in range(self.tamano_poblacion)]

        
        self.probabilidad_mutacion = 0.01
        self.probabilidad_cruce = 0.95
        self.metodo_seleccion = metodo_seleccion
        self.num_generaciones = 0
        self.mejor_individuo = None
        self.mejores_fitness = []
        self.mejor_individuo_total = None

        self.num_iter_converg_max = 3
        self.tolerancia = 0.05

    

    def crear_individuo(self, individuo_base, orden_estantes):
        orden_estantes = list(orden_estantes) # convertimos a lista para asegurar que es una lista
        nuevo_layout = [fila[:] for fila in individuo_base.layout]  # copiamos la matriz de layout

        for i, fila in enumerate(nuevo_layout):
            for j, estante in enumerate(fila):
                if estante != 0 and estante != -2 and orden_estantes: # verificamos que haya estantes disponibles
                    nuevo_layout[i][j] = orden_estantes.pop(0)

        return Individuo(nuevo_layout, individuo_base.lista_productos, individuo_base.ordenes_picking)

    
    def seleccion_ruleta(self): #CONVERGENCIA MAS LENTA PERO MAYOR EXPLORACION DEL ESPACIO DE RESULATADOS
        fitness_total = sum([individuo.fitness for individuo in self.individuos])
        probabilidades_seleccion = [individuo.fitness / fitness_total for individuo in self.individuos]
        
        # Selecciona dos padres aleatoriamente en base a las probabilidades de selección
        padre1 = np.random.choice(self.individuos, 1, p=probabilidades_seleccion)[0]
        padre2 = np.random.choice(self.individuos, 1, p=probabilidades_seleccion)[0]
        while np.array_equal(padre2, padre1):
          padre2 = np.random.choice(self.individuos, 1, p=probabilidades_seleccion)[0]
   
        return padre1, padre2
    
    def seleccion_torneo(self, k=8):
        competidores = random.sample(self.individuos, k)
        seleccionados = sorted(competidores, key=lambda x: x.fitness, reverse=True)[:2]
        return seleccionados

    def seleccionar_puntos_cruce(self):   #selecciona aleatoriamente dos indices para realizar el cruce
        punto1 = random.randint(0, self.tamano_poblacion - 1)
        punto2 = random.randint(0, self.tamano_poblacion - 1)
        while punto2 == punto1:
            punto2 = random.randint(0, self.tamano_poblacion - 1)
        return punto1, punto2

    def cruce_de_orden(self, padre1_estantes, padre2_estantes):                               #RECIBE LAS LISTAS DE LOS ESTANTES DE LOS PADRES
        # Elegir dos puntos aleatorios para el cruce
        punto1 = random.randint(0, len(padre1_estantes) - 1)
        punto2 = random.randint(0, len(padre2_estantes) - 1)

        # Ordenar los puntos para asegurarnos de que punto1 < punto2
        punto1, punto2 = min(punto1, punto2), max(punto1, punto2)

        # Obtener las sublistas de cada padre dentro de los puntos de cruce
        sublista1 = padre1_estantes[punto1:punto2]
        sublista2 = padre2_estantes[punto1:punto2]

        # Crear los descendientes intercambiando las sub-listas
        hijo1_estantes = [elem for elem in padre1_estantes if elem not in sublista2]
        hijo1_estantes[punto1:punto2] = sublista2
        

        hijo2_estantes = [elem for elem in padre2_estantes if elem not in sublista1]
        hijo2_estantes[punto1:punto2] = sublista1
        
        return hijo1_estantes, hijo2_estantes            #retorna listas  de estantes

    
    def mutacion_por_mezcla(self, individuo_lista):
        # Seleccionar genes al azar para mutar
        genes_a_mutar = random.sample(range(len(individuo_lista)), random.randint(1, len(individuo_lista)))

        # Mezclar los valores de los genes elegidos
        valores_a_mezclar = [individuo_lista[i] for i in genes_a_mutar]
        random.shuffle(valores_a_mezclar)

        # Crear una copia de la lista de orden_estantes mutada
        orden_estantes_mutada = individuo_lista.copy()

        # Reemplazar los valores de los genes elegidos con los valores mezclados
        for i in genes_a_mutar:
            orden_estantes_mutada[i] = valores_a_mezclar.pop(0)

        return orden_estantes_mutada

    
    def reemplazo(self, descendientes):
        self.individuos = descendientes[:len(self.individuos)]


    def determinar_mejor_individuo(self):
        mejor_individuo = self.individuos[0]
        for individuo in self.individuos:
            if individuo.fitness < mejor_individuo.fitness:
                mejor_individuo = individuo
                
        return mejor_individuo


    def evolucionar(self, prob_cruce=None, prob_mutacion=None, metodo_seleccion=None):
        # Actualizar los parámetros de la evolución si se especifican
        if prob_cruce is not None:
            self.probabilidad_cruce = prob_cruce
        if prob_mutacion is not None:
            self.probabilidad_mutacion = prob_mutacion
        if metodo_seleccion is not None:
            self.metodo_seleccion = metodo_seleccion

        # Crear la nueva población
        nueva_poblacion = []

        # Defino un individuo base para generar otros individuos
        individuo_base = self.individuos[0]

        # Defino el mejor individuo de la generación actual y total
        if self.mejor_individuo is None:
            self.mejor_individuo = self.determinar_mejor_individuo()
        if self.mejor_individuo_total is None or self.mejor_individuo.fitness > self.mejor_individuo_total.fitness:
            self.mejor_individuo_total = self.mejor_individuo
        #Lista de mejores individuos de cada generación
        self.mejores_fitness.append(self.mejor_individuo.fitness)

        # Repetir hasta que la nueva población tenga el mismo tamaño que la población actual
        while len(nueva_poblacion) < len(self.individuos):
            # Seleccionar dos padres
            if self.metodo_seleccion == 1:
                padre1, padre2 = self.seleccion_torneo()
            else:
                padre1, padre2 = self.seleccion_ruleta()

            # Cruzar los padres con una cierta probabilidad de cruce
            if random.random() < self.probabilidad_cruce:
                descendiente1, descendiente2 = self.cruce_de_orden(padre1.orden_estantes, padre2.orden_estantes)
            else:
                descendiente1, descendiente2 = padre1.orden_estantes, padre2.orden_estantes

            # Mutar los descendientes con una cierta probabilidad de mutación
            if random.random() < self.probabilidad_mutacion:
                descendiente1 = self.mutacion_por_mezcla(descendiente1)
            if random.random() < self.probabilidad_mutacion:
                descendiente2 = self.mutacion_por_mezcla(descendiente2)

            # Transformar los descendientes en objetos individuo
            descendiente1 = self.crear_individuo(individuo_base, descendiente1)
            descendiente2 = self.crear_individuo(individuo_base, descendiente2)

            nueva_poblacion.append(descendiente1)
            nueva_poblacion.append(descendiente2)

        # Actualizar la población y el número de generaciones
        self.individuos = nueva_poblacion
        self.num_generaciones += 1

        # Actualizar el mejor individuo de la generación
        self.mejor_individuo = self.determinar_mejor_individuo()

        # Criterio de convergencia
        iter_convergentes = 0
        # Si el mejor individuo de la generación es casi igual al mejor individuo total:
        if abs(self.mejor_individuo_total.fitness - self.mejor_individuo.fitness)/ self.mejor_individuo_total.fitness < self.tolerancia:
            iter_convergentes += 1
            if (self.mejor_individuo.fitness > self.mejor_individuo_total.fitness):
                self.mejor_individuo_total = self.mejor_individuo

        # Si el nuevo individuo es considerablemente mejor que el mejor individuo total:
        if((self.mejor_individuo.fitness - self.mejor_individuo_total.fitness) > (self.tolerancia * self.mejor_individuo_total.fitness)):
            self.mejor_individuo_total = self.mejor_individuo
            iter_convergentes = 0

        # Verificar si se alcanzó la convergencia
        if iter_convergentes >= self.num_iter_converg_max:
            print("Alcanzado criterio de detención por convergencia en la generación {}".format(self.num_generaciones))
            return True

class Impresora:
    def imprimir(self, lista_de_arreglos):
        # Inicializar la matriz vacía
        matriz = np.empty((0, len(lista_de_arreglos[0])), int)
        # Determinar el ancho de columna requerido para imprimir la matriz
        col_width = len(str(np.max(lista_de_arreglos))) + 1
        # Agregar cada arreglo como una fila a la matriz
        for arreglo in lista_de_arreglos:
            fila = np.array(arreglo).reshape((1, -1))
            matriz = np.vstack((matriz, fila))
        # Guardar la matriz en un archivo temporal
        with open('temp.txt', 'w') as f:
            np.savetxt(f, matriz, fmt='%{}d'.format(col_width), delimiter=' ')
        # Leer el archivo temporal y mostrar la matriz en el formato deseado
        with open('temp.txt', 'r') as f:
            for line in f:
                print(line.rstrip())
        
        # Eliminar el archivo temporal
        os.remove('temp.txt')
        
class Reporte:
    def __init__(self, poblacion):
        self.num_generaciones = poblacion.num_generaciones
        self.tam_poblacion = poblacion.tamano_poblacion
        self.prob_mutacion = poblacion.probabilidad_mutacion
        self.prob_cruce = poblacion.probabilidad_cruce
        self.metodo_seleccion = poblacion.metodo_seleccion
        self.mejor_individuo = poblacion.mejor_individuo


    def generar_reporte(self, tiempo_ejecucion, nombre_archivo):
        with open(nombre_archivo +'.txt', 'a') as f:
            f.write("Reporte de Ejecucion\n")
            f.write("--------------------\n")
            f.write(f"Mejor Individuo: {self.mejor_individuo.orden_estantes}\n")
            f.write(f"Fitness: {self.mejor_individuo.fitness}\n")
            f.write(f"Numero de Generaciones: {self.num_generaciones}\n")
            f.write(f"Tamano de Poblacion: {self.tam_poblacion}\n")
            f.write(f"Probabilidad de Mutacion: {self.prob_mutacion}\n")
            f.write(f"Probabilidad de Cruce: {self.prob_cruce}\n")
            f.write(f"Tipo de Seleccion: {self.metodo_seleccion}\n")
            f.write(f"Tiempo de Ejecucion: {round(tiempo_ejecucion, 3)} segundos\n")
            f.write("--------------------\n")


