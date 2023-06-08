import matplotlib.pyplot as plt
def graficar(x_entrenamiento,t_entrenamiento,x_prueba,t_prueba,loss_totales,EPOCHS):
        # Graficamos los datos de entrenamiento y prueba
        fig, axes = plt.subplots(1, 2)
        # Graficar el primer gráfico en el primer eje
        axes[0].scatter(x_entrenamiento[:, 0], x_entrenamiento[:, 1], c=t_entrenamiento)
        axes[0].set_title('Gráfico de entrenamiento')
        # Graficar el segundo gráfico en el segundo eje
        axes[1].scatter(x_prueba[:, 0], x_prueba[:, 1], c=t_prueba)
        axes[1].set_title('Gráfico de prueba')
        # Ajustar el espaciado entre los subplots
        plt.tight_layout()
        # Mostrar la figura
        plt.show()
        # Graficar el loss
        epochs_totales = list(range(1, EPOCHS+1))  # Rango de 1000 a 10000 con incrementos de 1000
        plt.plot(epochs_totales, loss_totales, 'm')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss durante el entrenamiento')
        plt.show()

