import numpy as np
#from layout_grafico import*

class Layout:
    def __init__(self, tam_estanteria, tam_pasillo, filas, columnas, tam_borde, bahia_carga):
        self.tam_estanteria = tam_estanteria
        self.tam_pasillo = tam_pasillo
        self.filas = filas
        self.columnas = columnas
        self.tam_borde = tam_borde
        self.bahia_carga = bahia_carga

        self.layout_matrix = None
        self.generate_layout()

    def generate_layout(self):
        # Crea la matriz de ceros de tamaño filas x columnas
        self.layout_matrix = np.zeros((self.filas, self.columnas), dtype=int)

        # Rellena la matriz con las estanterías
        for i in range(self.tam_borde, self.filas - self.tam_borde, self.tam_estanteria[0] + self.tam_pasillo):
            for j in range(self.tam_borde, self.columnas - self.tam_borde, self.tam_estanteria[1] + self.tam_pasillo):
                for k in range(self.tam_estanteria[0]):
                    for l in range(self.tam_estanteria[1]):
                        self.layout_matrix[i + k][j + l] = 1

        # Rellena el borde de la matriz con ceros
        self.layout_matrix[:self.tam_borde,:] = 0
        self.layout_matrix[-self.tam_borde:,:] = 0
        self.layout_matrix[:, :self.tam_borde] = 0
        self.layout_matrix[:, -self.tam_borde:] = 0

        # Agrega la casilla de carga y descarga en la posición indicada por el parámetro bahia_carga
        self.layout_matrix[self.bahia_carga[0], self.bahia_carga[1]] = -2



    def __repr__(self):
        # Ancho de cada columna en la matriz
        col_width = len(str(np.max(self.layout_matrix))) + 1
        # Imprime la matriz en un archivo temporal
        with open('temp.txt', 'w') as f:
            np.savetxt(f, self.layout_matrix, fmt='%{}d'.format(col_width), delimiter='')
        # Lee el archivo temporal y retorna su contenido como string
        with open('temp.txt', 'r') as f:
            return f.read()

"""
tam_estanteria = (3, 2)
tam_pasillo = 1
filas = 15
columnas = 12
tam_borde = 2

layout = Layout(tam_estanteria, tam_pasillo, filas, columnas, tam_borde, [2,0])
print(layout)
graficar_matriz(layout.layout_matrix)
"""
