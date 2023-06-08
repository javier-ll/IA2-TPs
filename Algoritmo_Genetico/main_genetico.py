from alg_genetico import*
from lector_ordenes import*
from layout_grafico import*
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Definir layout inicial
    tam_estanteria = (6, 2)
    tam_pasillo = 1
    filas = 24
    columnas = 12
    tam_borde = 2
    bahia_carga = [2, 0]

    layout_in = Layout(tam_estanteria, tam_pasillo, filas, columnas, tam_borde, bahia_carga)
    #print("layout inicial: \n", layout_in.layout_matrix)


    # Definir lista de productos
    lista_productos = list(range(1, 109))
    

    # Definir órdenes de picking
    #ordenes = generar_ordenes(25)
    ordenes = leer_ordenes("orders.txt")
   
    # Crear población
    tamano_poblacion = 15
    poblacion = Poblacion(tamano_poblacion, lista_productos, layout_in.layout_matrix, ordenes, 0)

    # Evolucionar la población
    num_generaciones = 100
    inicio = time.time()
    for i in range(num_generaciones):
        hilo = poblacion.evolucionar(0.95, 0.01, 1)
        if hilo==True:
            break
    fin = time.time()
    tiempo_ejecucion = round(fin - inicio, 3)

    # Imprimir el mejor individuo de la última generación
    #impresora = Impresora()
    #print("Mejor individuo de la última generación: \n")
    #impresora.imprimir(poblacion.mejor_individuo.layout)
    print("\n fitness: ", poblacion.mejor_individuo.fitness)
    graficar_matriz(poblacion.mejor_individuo.layout, poblacion.mejor_individuo.fitness)
    reporte = Reporte(poblacion)
    reporte.generar_reporte(tiempo_ejecucion, "reportes")
    

    print(poblacion.num_generaciones, poblacion.mejor_individuo.fitness)
    #plotear los fitness de los mejores individuos de cada generación vs generación
    plt.plot(list(range(1, poblacion.num_generaciones+1)), poblacion.mejores_fitness)
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Fitness de los mejores individuos de cada generación")
    plt.show()

