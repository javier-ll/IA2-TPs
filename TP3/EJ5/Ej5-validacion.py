import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    #return 0.5 * np.sin(3 * x)
    #return np.log10(abs(3*x))
    #return np.abs(0.4*x)
    return (np.exp(-x**2)) * x

def generar_datos_regresion(cantidad_ejemplos, funcion, rango_min=-8, rango_max=8):
    x = np.linspace(rango_min, rango_max, cantidad_ejemplos).reshape(-1, 1)
    amplitud = funcion(x)
    escala = np.random.uniform(0.85, 1.15, cantidad_ejemplos)
    t = amplitud * escala.reshape(-1, 1)
    return x, t

# Función para realizar el feed-forward de la red neuronal
def ejecutar_adelante(x, pesos):
    z = x.dot(pesos["w1"]) + pesos["b1"]
    h = np.maximum(0, z)                    # función de activación ReLU
    y = h.dot(pesos["w2"]) + pesos["b2"]
    return {"z": z, "h": h, "y": y}

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

# Función para realizar predicciones con la red neuronal entrenada
def hacer_regresion(x, pesos):
    resultados_feed_forward = ejecutar_adelante(x, pesos)
    y = resultados_feed_forward["y"]
    return y


# Función para entrenar la red neuronal
def train(x, t, x_val, t_val, pesos, learning_rate, epochs, validation_interval, tolerance):
    loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    it_list = []  # Lista para almacenar las iteraciones de validación

    for i in range(epochs):
        # feed-forward
        z, h, y = ejecutar_adelante(x, pesos).values()
        loss, d = regresion_loss(t, y)
        w1, b1, w2, b2 = pesos["w1"], pesos["b1"], pesos["w2"], pesos["b2"]

        # Retropropagación
        dL_dw2 = h.T.dot(d)
        dL_db2 = np.sum(d, axis=0, keepdims=True)
        dL_dh = d.dot(w2.T)
        dL_dz = dL_dh
        dL_dz[z <= 0] = 0

        dL_dw1 = x.T.dot(dL_dz)
        dL_db1 = np.sum(dL_dz, axis=0, keepdims=True)

        # Actualización de los pesos
        w1 -= learning_rate * dL_dw1
        b1 -= learning_rate * dL_db1
        w2 -= learning_rate * dL_dw2
        b2 -= learning_rate * dL_db2

        pesos["w1"], pesos["b1"], pesos["w2"], pesos["b2"] = w1, b1, w2, b2  # Actualización de los pesos en el diccionario

        # Verificación si es momento de evaluar el conjunto de validación
        if i % validation_interval == 0:
            y_val = hacer_regresion(x_val, pesos)
            val_loss, _ = regresion_loss(t_val, y_val)
            val_loss_history.append(val_loss)
            loss_history.append(loss)
            it_list.append(i)

            # Verificación de mejora en la pérdida de validación
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            elif abs(val_loss - best_val_loss) > tolerance:
                print("Entrenamiento detenido tempranamente")
                break

    return loss_history, val_loss_history, it_list


# Función principal para ejecutar el entrenamiento
def iniciar(numero_ejemplos, n_entrada, n_capa_oculta, epochs, learning_rate, graficar_datos=False, graficar_loss=False):
    x, t = generar_datos_regresion(numero_ejemplos, fun)
    x_val, t_val = generar_datos_regresion(int(numero_ejemplos/4), fun)

    np.random.seed(0) #semilla para números aleatorios
    pesos = inicializar_pesos(n_entrada, n_capa_oculta)

    # Entrenamiento
    loss_history, val_loss_history, it_list = train(x, t, x_val, t_val, pesos, learning_rate, epochs, validation_interval=10, tolerance=1e-3)
    #print("tam loss history: ", len(loss_history))
    #print("tam val loss history: ", len(val_loss_history))

    if graficar_datos:
        # Preddición con los datos de entrenamiento
        y = hacer_regresion(x, pesos)
        # Visualización de los datos de entrenamiento y la curva de regresión
        visualizar_datos(x, t, y)

    if graficar_loss:
        # Visualización de la evolución del error de regresión
        visualizar_loss(loss_history, val_loss_history, it_list)

    print("MSE final monocapa: ", loss_history[-1])
    return pesos


def visualizar_datos(x, t, y):
    fig, ax = plt.subplots()
    # Graficar los datos de entrenamiento como puntos dispersos
    ax.scatter(x, t)
    # Graficar la curva de regresión
    ax.plot(x, y.flatten(), c='red')  # Se utiliza y.flatten() para asegurar que sea una matriz unidimensional
    ax.set(xlabel='x', ylabel='y', title='Regresión del conjunto')
    plt.show()


def visualizar_loss(loss_history, val_loss_history, it_list):
    fig, ax = plt.subplots()
    ax.plot(it_list, loss_history, label='train')
    ax.plot(it_list, val_loss_history, label='validation')
    ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Function')
    ax.legend()
    plt.show()

def test(x_test, t_test, pesos):
    y_test = hacer_regresion(x_test, pesos)
    mse, _ = regresion_loss(t_test, y_test)
    visualizar_datos(x_test, t_test, y_test)

    return mse



# Main
pesos_entrenados = iniciar(numero_ejemplos=300, n_entrada=1, n_capa_oculta=200, epochs=10000, learning_rate=0.05, graficar_datos=True, graficar_loss=True)

x_test, t_test = generar_datos_regresion(85, fun)  # Generar datos de prueba
mse_test = test(x_test, t_test, pesos_entrenados)  # Llamar a la función de prueba con los pesos entrenados
print("MSE en el conjunto de prueba:", mse_test)