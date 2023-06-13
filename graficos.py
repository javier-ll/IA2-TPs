import matplotlib.pyplot as plt
def graficar(x_entrenamiento,t_entrenamiento,x_prueba,t_prueba):
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
def graficar_perdida_sin_parada(loss_entrenamiento_totales,EPOCHS):
        # Graficar la función de pérdida a lo largo del entrenamiento
        epochs_totales = list(range(1, EPOCHS+1))  # Rango de 1000 a 10000 con incrementos de 1000
        plt.plot(epochs_totales, loss_entrenamiento_totales, 'm')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss durante el entrenamiento')
        plt.show()
def graficar_perdida_con_parada(N,loss_entrenamiento_totales,loss_validacion_totales,EPOCHS):
        # Truncar el vector más largo al tamaño del vector más corto
        length = min(len(loss_entrenamiento_totales), len(loss_validacion_totales))
        loss_entrenamiento_totales = loss_entrenamiento_totales[:length]
        loss_validacion_totales = loss_validacion_totales[:length]
        # Graficar los dos vectores en el mismo gráfico
        plt.plot(loss_entrenamiento_totales, label='Entrenamiento')
        plt.plot(loss_validacion_totales, label='Validación')
        plt.legend()
        plt.show()



