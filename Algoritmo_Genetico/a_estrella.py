import math
import numpy as np
from layout4 import Layout
from temple2 import*

class Almacen:
    def __init__(self, matriz):
        self.matriz = matriz
        self.productos = {}
        for i, fila in enumerate(matriz):
            for j, valor in enumerate(fila):
                if valor > 0:
                    codigo_producto = str(valor)
                    self.productos[codigo_producto] = (i, j)

    def buscar_producto(self, codigo_producto):
        if codigo_producto in self.productos:
            return self.productos[codigo_producto]
        else:
            return None
    
    def buscar_punto_transitable_cerca(self, coordenada, radio):
        """Busca el punto transitable (valor 0) más cercano a una coordenada dada
        dentro de un radio determinado"""
        x, y = coordenada
        for r in range(1, radio+1):
            for i in range(x-r, x+r+1):
                for j in range(y-r, y+r+1):
                    if 0 <= i < len(self.matriz) and 0 <= j < len(self.matriz[0]):
                        if self.matriz[i][j] == 0 and math.sqrt((x-i)**2 + (y-j)**2) <= r:
                            return (i, j)
        return None



class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    



def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path
            break

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            if maze[current_node.position[0]][current_node.position[1]] == 1:
                # Found an obstacle, continue searching
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def a_estrella_test(matriz, start, end):
    path = astar(matriz, start, end)
    print(path)

def a_estrella_iniciar(matriz, start, end, ordenes):

    # Creamos un almacen
    almacen = Almacen(matriz)

    # variables
    num_ordenes = 0
    num_productos = 0
    total_productos = 0
    orden_actual = None
    productos_actual = []
    it = 0
    i=0
    nOrdenAnterior = 0
    coordenada_cercana_producto = 0
    productos = []
    productos_vector = []



    productos_actual = []
    for elem in ordenes:
        if elem.startswith(" "):
            continue
        elif elem.startswith("Order "):
            if it >= 1:
                print(f"Orden {orden_actual}: {num_productos} productos")
                productos_vector.append(productos_actual) # agregar lista de productos al vector de productos
            num_ordenes += 1
            orden_actual = elem
            num_productos = 0
            productos_actual = [] # crear nueva lista para la nueva orden
            it += 1
        elif elem == "$":
            productos_vector.append(productos_actual) # agregar última lista de productos al vector de productos
            break
        else:
            num_productos += 1
            elem = str(elem)
            elem = elem.replace("P", "")
            productos_actual.append(elem) # agregar producto a la lista actual
            
            #coordenadas_producto = almacen.buscar_producto(str(elem)) # busco el producto por codigo
            """
            if coordenadas_producto is not None:
                print("Producto ", elem ," OK en el almacén.")
            else:
                print("Producto ", elem ," NO se encontró en el almacén.")
                
                """
    print(f"Total de ordenes: {num_ordenes}")
    print(f"Total de productos: {total_productos}")  
    
    print("los productos son: ", productos)
    print("productos vector: ", productos_vector)


    print("Indique el numero de orden que desea analizar. Total:",num_ordenes)
    orden = input()

    productos = productos_vector[int(orden)-1]
    print("productos elegidos: ", productos)
    # Integracion con temple
    #Funcion para buscar productos por numero y retorna coordenadas

    # Crear una matriz para almacenar los costos
    costos = np.zeros((1, 3))
    costos = list(costos)
    #coordenadas_producto= list(coordenadas_producto)

    coordenadas_producto = []
    costo = 0
    costos = []
    # Iterar a través de cada par de productos y calcular el costo entre ellos
    index = 0
    for i in range(-1,num_productos):
        for j in range(num_productos):


            if i==-1:#inicio
                coordenadas_producto = almacen.buscar_producto(str(productos[j])) # busco el producto por codigo
                coordenada_cercana_producto = almacen.buscar_punto_transitable_cerca(coordenadas_producto,1)
                costo = astar(matriz,start, coordenada_cercana_producto)
                costos.append(['i', str(productos[j]), len(costo)])
            else:
                if(str(productos[i]) != str(productos[j]) and i!=-1 and i!=-2):
                    coordenadas_producto_A = almacen.buscar_producto(str(productos[i])) # busco el producto por codigo
                    coordenada_cercana_producto_A = almacen.buscar_punto_transitable_cerca(coordenadas_producto_A,1)
                    
                    coordenadas_producto_B = almacen.buscar_producto(str(productos[j])) # busco el producto por codigo
                    coordenada_cercana_producto_B = almacen.buscar_punto_transitable_cerca(coordenadas_producto_B,1)

                    costo = astar(matriz,coordenada_cercana_producto_A, coordenada_cercana_producto_B)
                    costos.append([str(productos[i]), str(productos[j]), len(costo)])


    # filtrado para evitar repetidos
    
    costosFinal = []
    for costo in costos:

        a = costo[1]
        b = costo[0]
        c = costo[2]
        costo2 = [a,b,c]

        if costo[0] == 'i':
            costosFinal.append(costo)
        else:
            if costo2 in costos and not costo2 in costosFinal:
                costosFinal.append(costo)    

    #if num_productos > 0:
    #    print(f"Orden {orden_actual}: {num_productos} productos")
    #    total_productos += num_productos

   
 
    return(costosFinal)






#----------------------------------     TESTEO  Almacen   ----------------------------------#
# Creo una matriz layout con mi clase Layout, le cargo los prodcutos tal como lo haré en el algoritmo genético 
# y se lo paso a la clase Almacen para que genere un objeto analogo pero compatible con el algoritmo A*. 
# Así es que a partir de mi layout completo se genera el diccionario de productos y sus coordenadas en el almacén. 

import random
def cargar_productos(layout_matrix, lista_productos):
        numeros_productos = list(range(1, len(lista_productos)+1))
        for i in range(len(layout_matrix)):
            for j in range(len(layout_matrix[0])):
                if layout_matrix[i][j] != 0 and layout_matrix[i][j] != -2:
                    producto = random.choice(numeros_productos)
                    layout_matrix[i][j] = producto
                    numeros_productos.remove(producto)


tam_estanteria = (3, 2)
tam_pasillo = 1
filas = 15
columnas = 12
tam_borde = 2

layout_vacio = Layout(tam_estanteria, tam_pasillo, filas, columnas, tam_borde, [2,0])
lista_productos = [i for i in range(1, 55)]
cargar_productos(layout_vacio.layout_matrix, lista_productos)
layout_full = layout_vacio       #actualizo el layout vacio con los productos
print("Layout: ")
print(layout_full.layout_matrix)


alm=Almacen(layout_full.layout_matrix)
#print("clase Almacen:")
#print(alm.matriz)
#print("productos:\n", alm.productos)

#--------------------------------------------------------------------------------------------#

#------------------------------------     TESTEO  A*   --------------------------------------#

mis_ordenes=["Order 1", "P10", "P17", "P22", "P25", "P31", "P39", "Order 2", "P17", "P23", "P30", "P37", "$"]
costos_prueba = a_estrella_iniciar(layout_full.layout_matrix, [2,0], [2,0], mis_ordenes)
print("costos_prueba: ", costos_prueba)
print("\n FIN COSTOS PRUEBA\n")

#---------------------------------------------------------------------------------------------#
costo, inicio = iniciar_temple(costos_prueba)
print("inicio: ", inicio)
print("costo: ", costo)