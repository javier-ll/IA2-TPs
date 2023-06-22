import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time

#--------------------------------------------------------------------------------------------------------------------------------------------------
#                                                          RED NEURONAL
#--------------------------------------------------------------------------------------------------------------------------------------------------
def fun(x):
    return 0.5 * np.sin(3 * x)

def generar_datos_regresion(cantidad_ejemplos, funcion, rango_min=-2, rango_max=2):
    x = np.linspace(rango_min, rango_max, cantidad_ejemplos).reshape(-1, 1)
    amplitud = funcion(x)
    escala = np.random.uniform(0.85, 1.15, cantidad_ejemplos)
    t = amplitud * escala.reshape(-1, 1)
    return x, t

def ejecutar_adelante(x, pesos, funcion_activacion):
    z = x.dot(pesos["w1"]) + pesos["b1"]
    h = funcion_activacion(z)
    y = h.dot(pesos["w2"]) + pesos["b2"]
    return {"z": z, "h": h, "y": y}

def relu(x):
    return np.maximum(0, x)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def regresion_loss(t, y):
    mse = np.mean((t - y) ** 2)
    deriv = -2 * (t - y) / len(t)
    return mse, deriv

def inicializar_pesos(n_entrada, n_capa_oculta):
    randomgen = np.random.default_rng()
    w1 = 0.1 * randomgen.standard_normal((n_entrada, n_capa_oculta))
    b1 = 0.1 * randomgen.standard_normal((1, n_capa_oculta))
    w2 = 0.1 * randomgen.standard_normal((n_capa_oculta, 1))
    b2 = 0.1 * randomgen.standard_normal((1, 1))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

def hacer_regresion(x, pesos, funcion_activacion):
    resultados_feed_forward = ejecutar_adelante(x, pesos, funcion_activacion)
    y = resultados_feed_forward["y"]
    return y

def train(x, t, pesos, learning_rate, epochs, funcion_activacion, funcion_activacion_derivada):
    loss_history = []

    for i in range(epochs):
        z, h, y = ejecutar_adelante(x, pesos, funcion_activacion).values()
        loss, d = regresion_loss(t, y)
        w1, b1, w2, b2 = pesos["w1"], pesos["b1"], pesos["w2"], pesos["b2"]

        dL_dw2 = h.T.dot(d)
        dL_db2 = np.sum(d, axis=0, keepdims=True)
        dL_dh = d.dot(w2.T)
        dL_dz = dL_dh * funcion_activacion_derivada(z)

        dL_dw1 = x.T.dot(dL_dz)
        dL_db1 = np.sum(dL_dz, axis=0, keepdims=True)

        w1 -= learning_rate * dL_dw1
        b1 -= learning_rate * dL_db1
        w2 -= learning_rate * dL_dw2
        b2 -= learning_rate * dL_db2

        pesos["w1"], pesos["b1"], pesos["w2"], pesos["b2"] = w1, b1, w2, b2

        if i % 100 == 0:
            loss_history.append(loss)

    return loss_history


def test(x_test, t_test, pesos, funcion_activacion):
    y_test = hacer_regresion(x_test, pesos, funcion_activacion)
    mse, _ = regresion_loss(t_test, y_test)
    return mse



#----------------------------------------------------------------------------------------------------------------------------------------------------
#                                                              ANÁLISIS DE HIPERPARÁMETROS
#----------------------------------------------------------------------------------------------------------------------------------------------------

def variar_hiperparametros(numero_ejemplos, n_entrada, n_capa_oculta, learning_rates, funciones_activacion, epochs):
    mse_resultados = {}  # Diccionario para almacenar los resultados de MSE
    x, t = generar_datos_regresion(numero_ejemplos, fun)  # Generar datos de entrenamiento
    
    print("Variando hiperparámetros...")
    for funcion_activacion in funciones_activacion:
        for cant_neuronas in n_capa_oculta:
            for learning_rate in learning_rates:
                for n_epochs in epochs:
                    mse_scores = []  # Lista para almacenar los puntajes de MSE en cada iteración

                    for _ in range(10):  # Realizar 10 iteraciones de K-Fold Cross Validation
                        kf = KFold(n_splits=10, shuffle=True)  # Crear objeto KFold para dividir los datos
                        mse_sum = 0

                        for train_index, val_index in kf.split(x):  # Iterar sobre las particiones de entrenamiento y validacion
                            x_train, x_val = x[train_index], x[val_index]  # Obtener los conjuntos de entrenamiento y validación
                            t_train, t_val = t[train_index], t[val_index]

                            pesos = inicializar_pesos(n_entrada, cant_neuronas)  # Inicializar los pesos de la red neuronal

                            # Seleccionar la función de activación y su derivada según el tipo de activación elegida
                            if funcion_activacion == "relu":
                                funcion_activacion_fn = relu
                                funcion_activacion_derivada = lambda x: np.where(x <= 0, 0, 1)
                            elif funcion_activacion == "sigmoide":
                                funcion_activacion_fn = sigmoide
                                funcion_activacion_derivada = lambda x: sigmoide(x) * (1 - sigmoide(x))

                            #print("Entrenando modelo...")
                            loss_history = train(x_train, t_train, pesos, learning_rate, n_epochs, funcion_activacion_fn,
                                                 funcion_activacion_derivada)  # Entrenar el modelo y obtener el historial de pérdidas
                            mse = test(x_val, t_val, pesos, funcion_activacion_fn)  # Evaluar el modelo en el conjunto de validacion
                            mse_sum += mse
                        mse_prom = mse_sum / 10  # Calcular el MSE promedio de las 10 iteraciones de K-Fold Cross Validation
                        mse_scores.append(mse_prom)  # Agregar el MSE promedio a la lista de puntajes

                    mse_std = np.std(mse_scores)  # Calcular la desviación estándar de los puntajes de MSE
                    #mse_resultados.append((learning_rate, cant_neuronas, funcion_activacion, n_epochs, mse_scores, mse_std))
                    mse_resultados[(learning_rate, cant_neuronas, funcion_activacion, n_epochs)] = [mse_scores, mse_std] # Agregar los resultados al diccionario


    return mse_resultados

def obtener_hiperparametros_optimos(resultados_barrido):
    resultados_barrido_ordenados = sorted(resultados_barrido.items(), key=lambda x: (x[1][0], x[1][1]))  # Ordenar el diccionario por MSE promedio y luego por desviación estándar
    mejores_hiper = resultados_barrido_ordenados[0][0]  # Obtener los hiperparámetros óptimos

    learning_rate_opt = mejores_hiper[0]
    neuronas_opt = mejores_hiper[1]
    activacion_opt = mejores_hiper[2]
    epochs_opt = mejores_hiper[3]

    return learning_rate_opt, neuronas_opt, activacion_opt, epochs_opt


import matplotlib.pyplot as plt

def plot_box_plots(resultados_barrido, parametros_optimos):
    # Filtrar resultados según los parámetros óptimos
    learning_rate_opt = parametros_optimos[0]
    neuronas_opt = parametros_optimos[1]
    activacion_opt = parametros_optimos[2]
    epochs_opt = parametros_optimos[3]

    # Filtrar según neuronas_opt, activacion_opt, epochs_opt y variar learning_rate
    learning_rates = []
    mse_valores = []
    for parametros, datos in resultados_barrido.items():
        if (
            parametros[1] == neuronas_opt and
            parametros[2] == activacion_opt and
            parametros[3] == epochs_opt
        ):
            learning_rates.append(parametros[0])
            mse_valores.append(datos[0])

    # Filtrar según learning_rate_opt, activacion_opt, epochs_opt y variar neuronas_opt
    neuronas = []
    mse_valores_neuronas = []
    for parametros, datos in resultados_barrido.items():
        if (
            parametros[0] == learning_rate_opt and
            parametros[2] == activacion_opt and
            parametros[3] == epochs_opt
        ):
            neuronas.append(parametros[1])
            mse_valores_neuronas.append(datos[0])

    # Filtrar según learning_rate_opt, activacion_opt, neuronas_opt y variar epochs_opt
    epochs = []
    mse_valores_epochs = []
    for parametros, datos in resultados_barrido.items():
        if (
            parametros[0] == learning_rate_opt and
            parametros[1] == neuronas_opt and
            parametros[2] == activacion_opt
        ):
            epochs.append(parametros[3])
            mse_valores_epochs.append(datos[0])

    # Crear figura y subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Graficar variación de learning_rate
    axs[0].boxplot(mse_valores)
    axs[0].set_xticklabels(learning_rates)
    axs[0].set_xlabel('Learning Rate')
    axs[0].set_ylabel('MSE')
    axs[0].set_title('Variación de MSE con respecto a Learning Rate')

    # Graficar variación de neuronas_opt
    axs[1].boxplot(mse_valores_neuronas)
    axs[1].set_xticklabels(neuronas)
    axs[1].set_xlabel('Neuronas')
    axs[1].set_ylabel('MSE')
    axs[1].set_title('Variación de MSE con respecto a Neuronas')

    # Graficar variación de epochs_opt
    axs[2].boxplot(mse_valores_epochs)
    axs[2].set_xticklabels(epochs)
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('MSE')
    axs[2].set_title('Variación de MSE con respecto a Epochs')

    # Ajustar espaciado entre subplots
    fig.tight_layout()

    plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------
#                                                            MAIN
#--------------------------------------------------------------------------------------------------------------------------------------------------
numero_ejemplos = 150
n_entrada = 1
n_capa_oculta = [50, 100, 150, 200]
learning_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
funciones_activacion = ["relu", "sigmoide"]
epochs = [1000, 5000, 10000]

inicio = time.time()
resultados_barrido = variar_hiperparametros(numero_ejemplos, n_entrada, n_capa_oculta, learning_rates, funciones_activacion, epochs)
learning_rate_opt, neuronas_opt, activacion_opt, epochs_opt = obtener_hiperparametros_optimos(resultados_barrido)

print("Hiperparámetros óptimos:")
print("Learning Rate: {}, Neuronas: {}, Función de Activación: {}, Epochs: {}".format(learning_rate_opt, neuronas_opt, activacion_opt, epochs_opt))

hiper_optimos = [learning_rate_opt, neuronas_opt, activacion_opt, epochs_opt]
plot_box_plots(resultados_barrido, hiper_optimos)

fin = time.time()
print("Tiempo de ejecución del barrido: {} segundos".format(fin - inicio))

# TEST
x_test, t_test = generar_datos_regresion(20, fun)         #conjunto de datos de prueba
pesos_opt = inicializar_pesos(n_entrada, neuronas_opt)

# Determinar la función de activación óptima
if activacion_opt == "relu":
    funcion_activacion_opt = relu
elif activacion_opt == "sigmoide":
    funcion_activacion_opt = sigmoide

# Entrenar el modelo con los hiperparámetros óptimos
train(x_test, t_test, pesos_opt, learning_rate_opt, epochs_opt, funcion_activacion_opt, lambda x: funcion_activacion_opt(x) * (1 - funcion_activacion_opt(x)))

# Evaluar el modelo en el conjunto de prueba
mse_test = test(x_test, t_test, pesos_opt, funcion_activacion_opt)
print("MSE en el conjunto de prueba: {}".format(mse_test))
