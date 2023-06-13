import numpy as np
#from MLP_Clasificacion import ejecutar_adelante

def train_parada_temprana(x_entrenamiento, t_entrenamiento, x_validacion, t_validacion, pesos, learning_rate, epochs, paciencia, tolerancia, N):
    m_entrenamiento = np.size(x_entrenamiento, 0)
    m_validacion = np.size(x_validacion, 0)
    loss_entrenamiento_totales = []
    loss_validacion_totales = []
    mejor_loss_validacion = np.inf
    mejores_pesos = None
    epochs_sin_mejora = 0
    
    for i in range(epochs):
        resultados_feed_forward_entrenamiento = ejecutar_adelante(x_entrenamiento, pesos)
        y_entrenamiento = resultados_feed_forward_entrenamiento["y"]
        h_entrenamiento = resultados_feed_forward_entrenamiento["h"]
        z_entrenamiento = resultados_feed_forward_entrenamiento["z"]
        
        # Cálculo de la función de pérdida en el conjunto de entrenamiento
        loss_entrenamiento,p_entrenamiento=calcular_loss(y_entrenamiento,m_entrenamiento,t_entrenamiento)
        loss_entrenamiento_totales.append(loss_entrenamiento)
        
        if (i + 1) % N == 0:
            # Cálculo de la función de pérdida en el conjunto de validación
            resultados_feed_forward_validacion = ejecutar_adelante(x_validacion, pesos)
            y_validacion = resultados_feed_forward_validacion["y"]
            loss_validacion,p_validacion=calcular_loss(y_validacion,m_validacion,t_validacion)
            loss_validacion_totales.append(loss_validacion)
            
            # Verificación de la parada temprana
            if loss_validacion < mejor_loss_validacion - tolerancia:
                mejor_loss_validacion = loss_validacion
                mejores_pesos = pesos
                epochs_sin_mejora = 0
            else:
                epochs_sin_mejora += 1
                if epochs_sin_mejora >= paciencia:
                    print("La parada temprana se activó en el epoch ", i)
                    break
        
        # Ajuste de los pesos mediante retropropagación
        dL_dy_entrenamiento = p_entrenamiento
        dL_dy_entrenamiento[range(m_entrenamiento), t_entrenamiento] -= 1
        dL_dy_entrenamiento /= m_entrenamiento
        
        dL_dw2_entrenamiento = h_entrenamiento.T.dot(dL_dy_entrenamiento)
        dL_db2_entrenamiento = np.sum(dL_dy_entrenamiento, axis=0, keepdims=True)
        
        dL_dh_entrenamiento = dL_dy_entrenamiento.dot(pesos["w2"].T)
        dL_dz_entrenamiento = dL_dh_entrenamiento
        dL_dz_entrenamiento[z_entrenamiento <= 0] = 0
        
        dL_dw1_entrenamiento = x_entrenamiento.T.dot(dL_dz_entrenamiento)
        dL_db1_entrenamiento = np.sum(dL_dz_entrenamiento, axis=0)
        
        pesos["w1"] -= learning_rate * dL_dw1_entrenamiento
        pesos["b1"] -= learning_rate * dL_db1_entrenamiento
        pesos["w2"] -= learning_rate * dL_dw2_entrenamiento
        pesos["b2"] -= learning_rate * dL_db2_entrenamiento
    
    return mejores_pesos,pesos, loss_entrenamiento_totales, loss_validacion_totales

def ejecutar_adelante(x, pesos):
    # Funcion de entrada (a.k.a. "regla de propagacion") para la primera capa oculta
    z = x.dot(pesos["w1"]) + pesos["b1"]

    # Funcion de activacion ReLU para la capa oculta (h -> "hidden")
    h = np.maximum(0, z)

    # Salida de la red (funcion de activacion lineal). Esto incluye la salida de todas
    # las neuronas y para todos los ejemplos proporcionados
    y = h.dot(pesos["w2"]) + pesos["b2"]

    return {"z": z, "h": h, "y": y}

def calcular_loss(y,m,t):
    exp_scores = np.exp(y)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    p = exp_scores / sum_exp_scores
    loss = (1 / m) * np.sum(-np.log(p[range(m), t]))
    return loss,p
