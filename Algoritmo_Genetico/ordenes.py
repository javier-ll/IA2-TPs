import random

"""
# definir el layout
layout = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0, 9, 10, 0],
          [0, 11, 12, 0, 13, 14, 0, 15, 16, 0, 17, 18, 0, 19, 20, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 21, 22, 0, 23, 24, 0, 25, 26, 0, 27, 28, 0, 29, 30, 0, 0],
          [0, 31, 32, 0, 33, 34, 0, 35, 36, 0, 37, 38, 0, 39, 40, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
"""

def generar_orden(cantidad_productos):
    productos = random.sample(range(1, 55), cantidad_productos)
    return productos

def generar_ordenes(n):
    ordenes = []
    for i in range(n):
        cantidad_productos = random.randint(15, 20)
        orden = generar_orden(cantidad_productos)
        ordenes.append(orden)
    return ordenes

def calcular_costo(layout, orden):
    costo = 0
    for i in range(len(orden)-1):
        pos1 = get_posicion_producto(layout, orden[i])
        pos2 = get_posicion_producto(layout, orden[i+1])
        costo += abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    return costo

def get_posicion_producto(layout, producto):
    for i in range(len(layout)):
        for j in range(len(layout[i])):
            if layout[i][j] == producto:
                return (i,j)
    return None



"""
ordenes = generar_ordenes(10)

for orden in ordenes:
    costo = calcular_costo(layout, orden)
    resultado = [orden, costo]
    print(resultado)
"""