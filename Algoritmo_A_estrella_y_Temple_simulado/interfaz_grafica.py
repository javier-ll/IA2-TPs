import pygame
from a_estrella import a_estrella

def interfaz(matriz, inicio,ruta=None, almacen=None):
    # definir colores
    BLACK = (0, 0, 0)
    WHITE = (230, 230, 230)
    RED = (255,0,0)
    GREEN = (0, 255, 0)
    BLUE = (0,0,255)
    GREY = (155,155,155)
    
    # Tamaño de ma matriz que repecenta al almacen
    m = (matriz.filas)
    n = (matriz.columnas)
    tam_casilla =26
    objetivo= None

    # definir tamaño de ventana y tamaño de celda
    CELL_SIZEy = tam_casilla
    CELL_SIZEx = tam_casilla
    WINDOW_SIZE = (n*tam_casilla,m*tam_casilla )
  
    # crear matriz de ejemplo
    """ matriz = [[0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0]]"""
    # inicializar Pygame
    pygame.init()

    # crear ventana
    screen = pygame.display.set_mode(WINDOW_SIZE)

    # definir fuente
    font = pygame.font.SysFont('arial.ttf', 15)

        
    # Variable de bandera para detener el programa
    done = False

    # bucle principal
    while not done:
         # Manejar eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Establecer la variable de bandera como Verdadera
                done = True

        # manejar eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # dibujar fondo
        screen.fill(WHITE)

        # dibujar celdas
        for row in range(m):
            for col in range(n):
                # calcular coordenadas de celda
                x = col * CELL_SIZEx
                y = row * CELL_SIZEy
                # dibujar rectángulo
                if matriz.layout_matrix[row][col] == 1:
                    pygame.draw.rect(screen, BLACK, [x, y, CELL_SIZEx, CELL_SIZEy])
                    pygame.draw.rect(screen, WHITE, [x, y, CELL_SIZEx -1, CELL_SIZEy -1], 1)  # dibujar contorno      
                else:
                    pygame.draw.rect(screen, WHITE, [x, y, CELL_SIZEx, CELL_SIZEy])
                    pygame.draw.rect(screen, BLACK, [x, y, CELL_SIZEx, CELL_SIZEy], 1)  # dibujar contorno
                # dibujar número de fila y columna
                text = font.render(f"{row},{col}", True, RED)
                screen.blit(text, [x + 5, y + 5])


       # dibujar rutas
    
        if ruta is not None:
            
            for i in range(len(ruta)):
                
                if ruta[i] == "i":
                    x = inicio[1]
                    y = inicio[0]
                else:
                    x,y = almacen.buscar_producto(ruta[i])
                
                color = BLACK
                x = x * CELL_SIZEx + CELL_SIZEx // 2
                y = y * CELL_SIZEy + CELL_SIZEy // 2
                
                if i > 0:
                    pygame.draw.circle(screen, BLUE, [y, x], min(CELL_SIZEx, CELL_SIZEy) // 2)
                    # dibujar número de PROD
                text = font.render(f"{i}", True, GREEN)
                screen.blit(text, [y-3, x-3])

          
        # dibujar punto objetivo
        if objetivo is not None:
            x = objetivo[1] * CELL_SIZEx + CELL_SIZEx // 2
            y = objetivo[0] * CELL_SIZEy + CELL_SIZEy // 2
            pygame.draw.circle(screen, BLUE, [x, y], min(CELL_SIZEx, CELL_SIZEy) // 2)
        
        if inicio is not None:
            x = inicio[1] * CELL_SIZEx + CELL_SIZEx // 2
            y = inicio[0] * CELL_SIZEy + CELL_SIZEy // 2
            pygame.draw.circle(screen, GREEN, [x, y], min(CELL_SIZEx, CELL_SIZEy) // 2)

        # actualizar pantalla
        pygame.display.update()
    
    # Salir de Pygame
    pygame.quit()