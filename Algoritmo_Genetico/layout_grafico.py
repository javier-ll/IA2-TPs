import pygame

# Definimos los colores que usaremos
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

def graficar_matriz(matriz, fitness):
    # Definimos el tamaño de cada casilla y de la ventana
    tam_casilla = 38
    margen = 1
    filas = len(matriz)
    columnas = len(matriz[0])
    ancho_ventana = columnas * (tam_casilla + margen) + margen
    alto_ventana = filas * (tam_casilla + margen) 

    # Inicializamos Pygame
    pygame.init()

    # Creamos la ventana
    ventana = pygame.display.set_mode((ancho_ventana, alto_ventana))

    # Rellenamos la ventana de blanco
    ventana.fill(BLANCO)

    # Dibujamos la matriz
    for fila in range(filas):
        for columna in range(columnas):
            if matriz[fila][columna] == 0:
                # Casilla blanca
                pygame.draw.rect(ventana, BLANCO, [(margen + tam_casilla) * columna + margen, (margen + tam_casilla) * fila + margen, tam_casilla, tam_casilla])
            else:
                # Casilla gris con el número correspondiente
                pygame.draw.rect(ventana, NEGRO, [(margen + tam_casilla) * columna + margen, (margen + tam_casilla) * fila + margen, tam_casilla, tam_casilla])
                fuente = pygame.font.SysFont('Arial', 20)
                texto = fuente.render(str(matriz[fila][columna]), True, BLANCO)
                ventana.blit(texto, [(margen + tam_casilla) * columna + margen + 10, (margen + tam_casilla) * fila + margen + 10])

     # Mostramos el valor del fitness
    fuente = pygame.font.SysFont('Arial', 30)
    texto = fuente.render('Fitness: ' + str(fitness), True, NEGRO)
    ventana.blit(texto, [ancho_ventana // 2 - 75, alto_ventana - 40])

  
    # Actualizamos la ventana
    pygame.display.update()

    # Esperamos a que el usuario cierre la ventana
    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                return
