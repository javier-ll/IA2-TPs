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

def ejecutar_adelante(x, pesos):
    z1 = x.dot(pesos["w1"]) + pesos["b1"]
    h1 = np.maximum(0, z1)
    z2 = h1.dot(pesos["w2"]) + pesos["b2"]
    h2 = np.maximum(0, z2)
    y = h2.dot(pesos["w3"]) + pesos["b3"]
    return {"z1": z1, "h1": h1, "z2": z2, "h2": h2, "y": y}

def regresion_loss(t, y):
    mse = np.mean((t - y) ** 2)
    deriv = -2 * (t - y) / len(t)
    return mse, deriv

def inicializar_pesos(n_entrada, n_capa_oculta1, n_capa_oculta2, n_salida):
    randomgen = np.random.default_rng()
    w1 = 0.1 * randomgen.standard_normal((n_entrada, n_capa_oculta1))
    b1 = 0.1 * randomgen.standard_normal((1, n_capa_oculta1))
    w2 = 0.1 * randomgen.standard_normal((n_capa_oculta1, n_capa_oculta2))
    b2 = 0.1 * randomgen.standard_normal((1, n_capa_oculta2))
    w3 = 0.1 * randomgen.standard_normal((n_capa_oculta2, n_salida))
    b3 = 0.1 * randomgen.standard_normal((1, n_salida))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}

def hacer_regresion(x, pesos):
    resultados_feed_forward = ejecutar_adelante(x, pesos)
    y = resultados_feed_forward["y"]
    return y

def train(x, t, x_val, t_val, pesos, learning_rate, epochs, validation_interval, tolerance):
    loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    it_list = []
    
    for i in range(epochs):
        z1, h1, z2, h2, y = ejecutar_adelante(x, pesos).values()
        loss, d = regresion_loss(t, y)
        
        w1, b1, w2, b2, w3, b3 = pesos["w1"], pesos["b1"], pesos["w2"], pesos["b2"], pesos["w3"], pesos["b3"]
        
        dL_dh2 = d.dot(w3.T)
        dL_dz2 = dL_dh2.copy()
        dL_dz2[z2 <= 0] = 0
        
        dL_dh1 = dL_dz2.dot(w2.T)
        dL_dz1 = dL_dh1.copy()
        dL_dz1[z1 <= 0] = 0
        
        dL_dw3 = h2.T.dot(d)
        dL_db3 = np.sum(d, axis=0, keepdims=True)
        
        dL_dw2 = h1.T.dot(dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        dL_dw1 = x.T.dot(dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        w1 -= learning_rate * dL_dw1
        b1 -= learning_rate * dL_db1
        w2 -= learning_rate * dL_dw2
        b2 -= learning_rate * dL_db2
        w3 -= learning_rate * dL_dw3
        b3 -= learning_rate * dL_db3
        
        pesos["w1"], pesos["b1"], pesos["w2"], pesos["b2"], pesos["w3"], pesos["b3"] = w1, b1, w2, b2, w3, b3
        
        if i % validation_interval == 0:
            y_val = hacer_regresion(x_val, pesos)
            val_loss, _ = regresion_loss(t_val, y_val)
            val_loss_history.append(val_loss)
            loss_history.append(loss)
            it_list.append(i)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            elif abs(val_loss - best_val_loss) > tolerance:
                print("Entrenamiento detenido tempranamente")
                break

    print("MSE final dos capas:", loss_history[-1])
    return loss_history, val_loss_history, it_list        


def iniciar(numero_ejemplos, n_entrada, n_capa_oculta1, n_capa_oculta2, n_salida, epochs, learning_rate, graficar_datos=False, graficar_loss=False):
    x, t = generar_datos_regresion(numero_ejemplos, fun)
    x_val, t_val = generar_datos_regresion(int(numero_ejemplos/4), fun)
    
    np.random.seed(0)
    pesos = inicializar_pesos(n_entrada, n_capa_oculta1, n_capa_oculta2, n_salida)

    loss_history, val_loss_history, it_list = train(x, t, x_val, t_val, pesos, learning_rate, epochs, validation_interval=10, tolerance=1e-3)

    if graficar_datos:
        y = hacer_regresion(x, pesos)
        visualizar_datos(x, t, y)

    if graficar_loss:
        visualizar_loss(loss_history, val_loss_history, it_list)

    print("MSE final: ", loss_history[-1])
    return pesos

def visualizar_datos(x, t, y):
    fig, ax = plt.subplots()
    ax.scatter(x, t)
    ax.plot(x, y.flatten(), c='red')
    ax.set(xlabel='x', ylabel='y', title='Regresión del conjunto de entrenamiento')
    plt.show()

def visualizar_loss(loss_history, val_loss_history, it_list):
    fig, ax = plt.subplots()
    ax.plot(it_list, loss_history, label='Conjunto de entrenamiento')
    ax.plot(it_list, val_loss_history, label='Conjunto de validación')
    ax.set(xlabel='Iteraciones', ylabel='Loss', title='Evolución del loss')
    ax.legend()
    plt.show()

pesos_entrenados = iniciar(numero_ejemplos=300, n_entrada=1, n_capa_oculta1=200, n_capa_oculta2=200, n_salida=1, epochs=10000, learning_rate=0.05, graficar_datos=True, graficar_loss=True)
