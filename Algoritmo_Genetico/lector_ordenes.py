def leer_ordenes(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        lineas = f.readlines()
    ordenes = []
    orden_actual = []
    for linea in lineas:
        if linea.startswith("Order"):
            if orden_actual:
                ordenes.append(orden_actual)
            orden_actual = []
        elif linea.strip():
            producto = int(linea.strip().replace("P", ""))
            orden_actual.append(producto)
    if orden_actual:
        ordenes.append(orden_actual)
    return ordenes

def calcular_costo(layout, orden):
    costo = 0
    pos_anterior = get_posicion_producto(layout, -2)
    for i in range(len(orden)):
        pos_actual = get_posicion_producto(layout, orden[i])
        costo += abs(pos_anterior[0] - pos_actual[0]) + abs(pos_anterior[1] - pos_actual[1])
        pos_anterior = pos_actual
    pos_final = get_posicion_producto(layout, -2)
    costo += abs(pos_anterior[0] - pos_final[0]) + abs(pos_anterior[1] - pos_final[1])
    return costo


def get_posicion_producto(layout, producto):
    for i in range(len(layout)):
        for j in range(len(layout[i])):
            if layout[i][j] == producto:
                return (i,j)
    return None


"""
# definir el layout
layout = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8, 0, 9, 10, 0],
          [-2, 11, 12, 0, 13, 14, 0, 15, 16, 0, 17, 18, 0, 19, 20, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 21, 22, 0, 23, 24, 0, 25, 26, 0, 27, 28, 0, 29, 30, 0, 0],
          [0, 31, 32, 0, 33, 34, 0, 35, 36, 0, 37, 38, 0, 39, 40, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
print("layout: \n")
for fila in layout:
    print(fila)
"""