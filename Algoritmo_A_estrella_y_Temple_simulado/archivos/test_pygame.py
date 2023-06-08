import pygame

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
win_width = 800
win_height = 600
win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Menú")

# Definir los colores
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# Definir la clase de botón
class Button():
    def __init__(self, x, y, width, height, text, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color

    # Dibujar el botón
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))
        font = pygame.font.SysFont("comicsans", 30)
        text = font.render(self.text, 1, black)
        win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))

    # Chequear si el botón fue clickeado
    def is_clicked(self, pos):
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                return True
        return False

# Definir los botones
button1 = Button(50, 50, 200, 100, "Botón 1", red)
button2 = Button(50, 200, 200, 100, "Botón 2", green)
button3 = Button(50, 350, 200, 100, "Botón 3", blue)
button4 = Button(50, 500, 200, 100, "Botón 4", black)

# Función para dibujar el menú
def draw_menu():
    win.fill(white)
    button1.draw(win)
    button2.draw(win)
    button3.draw(win)
    button4.draw(win)
    pygame.display.update()

# Loop principal
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if button1.is_clicked(pos):
                print("Clickeaste el botón 1")
            elif button2.is_clicked(pos):
                print("Clickeaste el botón 2")
            elif button3.is_clicked(pos):
                print("Clickeaste el botón 3")
            elif button4.is_clicked(pos):
                print("Clickeaste el botón 4")

    draw_menu()

# Salir de Pygame
pygame.quit()
