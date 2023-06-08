import pandas as pd
import matplotlib.pyplot as plt


# Crear un diccionario vacío para almacenar los datos
data = {'mejor_individuo': [], 'fitness': [], 'num_generaciones': [], 'tam_poblacion': [], 'prob_mutacion': [],
        'prob_cruce': [], 'tipo_seleccion': [], 'tiempo_ejecucion': []}

# Abrir el archivo y leer cada línea
with open('reportes.txt', 'r') as f:
    lines = f.readlines()

# Iterar por cada línea y extraer la información relevante
for i in range(len(lines)):
    if 'Mejor Individuo:' in lines[i]:
        data['mejor_individuo'].append([int(x) for x in lines[i].split('[')[1].split(']')[0].split(',')])
    elif 'Fitness:' in lines[i]:
        data['fitness'].append(float(lines[i].split(': ')[1]))
    elif 'Numero de Generaciones:' in lines[i]:
        data['num_generaciones'].append(int(lines[i].split(': ')[1]))
    elif 'Tama' in lines[i]:
        data['tam_poblacion'].append(int(lines[i].split(': ')[1]))
    elif 'Probabilidad de Mutacion:' in lines[i]:
        data['prob_mutacion'].append(float(lines[i].split(': ')[1]))
    elif 'Probabilidad de Cruce:' in lines[i]:
        data['prob_cruce'].append(float(lines[i].split(': ')[1]))
    elif 'Tipo de Seleccion:' in lines[i]:
        data['tipo_seleccion'].append(int(lines[i].split(': ')[1]))
    elif 'Tiempo de Ejecucion:' in lines[i]:
        data['tiempo_ejecucion'].append(float(lines[i].split(': ')[1].split(' segundos')[0]))

# Crear el dataframe
df_reportes = pd.DataFrame(data)
print(df_reportes)

# Obtener una lista de los parámetros
parametros = ['prob_mutacion', 'prob_cruce', 'tam_poblacion', 'tipo_seleccion']

# Crear una lista vacía para almacenar los dataframes filtrados
df_param_filtrados = []

# Iterar por cada parámetro y filtrar los reportes
for parametro in parametros:
    # Crear una lista vacía para almacenar los dataframes filtrados
    df_param_filtrados = []
    # Obtener los valores únicos del parámetro
    valores_parametro = df_reportes[parametro].unique()
    # Si solo hay un valor, no es necesario filtrar
    if len(valores_parametro) == 1:
        df_param_filtrados.append(df_reportes)
    else:
        # Iterar por cada valor del parámetro
        for valor in valores_parametro:
            # Filtrar los reportes en los que el parámetro tiene ese valor
            df_filtrado = df_reportes[df_reportes[parametro] == valor]
            # Si solo hay un valor de otro parámetro, se agrega a la lista de dataframes filtrados
            for otro_parametro in parametros:
                if otro_parametro != parametro:
                    valores_otro_parametro = df_filtrado[otro_parametro].unique()
                    if len(valores_otro_parametro) == 1:
                        df_param_filtrados.append(df_filtrado)
                        break
            # Si hay varios valores de otro parámetro, se iterará por ellos en el próximo ciclo
    # Concatenar los dataframes filtrados
    df_filtrados_concatenados = pd.concat(df_param_filtrados)





#------------------------------------------------------  TIEPO DE EJECUCIÓN  ------------------------------------------------------#
# CRUCE VS TIEMPO DE EJECUCIÓN
# Filtrar los datos para los valores de prob_mutacion seleccionados
prob_mutacion_seleccionados = [0.1, 0.2]
df_filtrados = df_filtrados_concatenados[df_filtrados_concatenados['prob_mutacion'].isin(prob_mutacion_seleccionados)]

# Crear la figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Iterar por cada valor único de prob_mutacion y graficar en el primer subplot
for prob_mutacion in prob_mutacion_seleccionados:
    # Filtrar los datos para este valor de prob_mutacion
    df_filtrado = df_filtrados[df_filtrados['prob_mutacion'] == prob_mutacion]

    # Calcular el promedio de tiempo de ejecución para cada valor de prob_cruce
    promedios = df_filtrado.groupby('prob_cruce')['tiempo_ejecucion'].mean()

    # Ordenar los valores de prob_cruce y los promedios en orden creciente
    promedios_ordenados = promedios.sort_index()
    prob_cruce_vals = promedios_ordenados.index.values

    # Graficar la función para este valor de prob_mutacion en el primer subplot
    ax1.plot(prob_cruce_vals, promedios_ordenados, label=f'Prob Mutacion: {prob_mutacion}')

# Configurar la leyenda y los ejes del primer subplot
ax1.legend()
ax1.set_xlabel('Prob Cruce')
ax1.set_ylabel('Tiempo de Ejecución (s)')


# MUTACIÓN VS TIEMPO DE EJECUCIÓN
# Iterar por cada valor único de prob_cruce y graficar en el segundo subplot
prob_cruce_seleccionados = [0.5, 0.7]
df_filtrados = df_filtrados_concatenados[df_filtrados_concatenados['prob_cruce'].isin(prob_cruce_seleccionados)]
prob_cruce_vals = df_filtrados['prob_cruce'].unique()
for prob_cruce in prob_cruce_vals:
    # Filtrar los datos para este valor de prob_cruce
    df_filtrado = df_filtrados[df_filtrados['prob_cruce'] == prob_cruce]

    # Calcular el promedio de tiempo de ejecución para cada valor de prob_mutacion
    promedios = df_filtrado.groupby('prob_mutacion')['tiempo_ejecucion'].mean()

    # Ordenar los valores de prob_mutacion y los promedios en orden creciente
    promedios_ordenados = promedios.sort_index()
    prob_mutacion_vals = promedios_ordenados.index.values

    # Graficar la función para este valor de prob_cruce en el segundo subplot
    ax2.plot(prob_mutacion_vals, promedios_ordenados, label=f'Prob Cruce: {prob_cruce}')

# Configurar la leyenda y los ejes del segundo subplot
ax2.legend()
ax2.set_xlabel('Prob Mutacion')
ax2.set_ylabel('Tiempo de Ejecución (s)')

#--------------------------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------  FITNESS  ---------------------------------------------------------#
# CRUCE VS TIEMPO DE EJECUCIÓN
# Filtrar los datos para los valores de prob_mutacion seleccionados
prob_mutacion_seleccionados = [0.1, 0.2]
df_filtrados = df_filtrados_concatenados[df_filtrados_concatenados['prob_mutacion'].isin(prob_mutacion_seleccionados)]

# Crear la figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Iterar por cada valor único de prob_mutacion y graficar en el primer subplot
for prob_mutacion in prob_mutacion_seleccionados:
    # Filtrar los datos para este valor de prob_mutacion
    df_filtrado = df_filtrados[df_filtrados['prob_mutacion'] == prob_mutacion]

    # Calcular el promedio de tiempo de ejecución para cada valor de prob_cruce
    promedios = df_filtrado.groupby('prob_cruce')['fitness'].mean()

    # Ordenar los valores de prob_cruce y los promedios en orden creciente
    promedios_ordenados = promedios.sort_index()
    prob_cruce_vals = promedios_ordenados.index.values

    # Graficar la función para este valor de prob_mutacion en el primer subplot
    ax1.plot(prob_cruce_vals, promedios_ordenados, label=f'Prob Mutacion: {prob_mutacion}')

# Configurar la leyenda y los ejes del primer subplot
ax1.legend()
ax1.set_xlabel('Prob Cruce')
ax1.set_ylabel('Fitness')


# MUTACIÓN VS TIEMPO DE EJECUCIÓN
# Iterar por cada valor único de prob_cruce y graficar en el segundo subplot
prob_cruce_seleccionados = [0.5, 0.7]
df_filtrados = df_filtrados_concatenados[df_filtrados_concatenados['prob_cruce'].isin(prob_cruce_seleccionados)]
prob_cruce_vals = df_filtrados['prob_cruce'].unique()
for prob_cruce in prob_cruce_vals:
    # Filtrar los datos para este valor de prob_cruce
    df_filtrado = df_filtrados[df_filtrados['prob_cruce'] == prob_cruce]

    # Calcular el promedio de tiempo de ejecución para cada valor de prob_mutacion
    promedios = df_filtrado.groupby('prob_mutacion')['fitness'].mean()

    # Ordenar los valores de prob_mutacion y los promedios en orden creciente
    promedios_ordenados = promedios.sort_index()
    prob_mutacion_vals = promedios_ordenados.index.values

    # Graficar la función para este valor de prob_cruce en el segundo subplot
    ax2.plot(prob_mutacion_vals, promedios_ordenados, label=f'Prob Cruce: {prob_cruce}')

# Configurar la leyenda y los ejes del segundo subplot
ax2.legend()
ax2.set_xlabel('Prob Mutacion')
ax2.set_ylabel('Fitness')

#--------------------------------------------------------------------------------------------------------------------------------#

plt.show()
