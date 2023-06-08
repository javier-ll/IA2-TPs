import random

n = 10  # tamaño de la matriz
m = 2   # tamaño de la góndola

# crear matriz
matriz = [[0 for i in range(n)] for j in range(n)]

# crear pasillos horizontales y verticales
for i in range(n):
    for j in range(n):
        if i % (m + 1) == 0 or j % (m + 1) == 0:
            matriz[i][j] = 1

# colocar productos en góndolas
for i in range(0, n, m + 1):
    for j in range(0, n, m + 1):
        if i + m < n and j + m < n:
            for k in range(i, i + m):
                for l in range(j, j + m):
                    matriz[k][l] = random.randint(0, 1)

# mostrar matriz
for row in matriz:
    print(row)
