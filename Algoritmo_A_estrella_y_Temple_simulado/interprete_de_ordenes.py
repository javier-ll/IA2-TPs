
import csv

def leer_ordenenes():
    ordenes = []
    orden = []
    nOrden = 0

    # Abrir el archivo CSV
    with open('orders.csv', 'r') as archivo_csv:
        # Crear un objeto csv.reader para leer el archivo
        lector_csv = csv.reader(archivo_csv)
        
        # Iterar sobre las filas del archivo
        for fila in lector_csv:
            # Procesar la fila según sea necesario
            valor = str(fila)
            valor = valor.replace("[", "")
            valor = valor.replace("]", "")
            valor = valor.replace("'", "")
            valor = valor.replace("'", "")
            ordenes.append(valor)

    ordenes.append("$") # caracter para identificar el final
    return ordenes

def ver_Ordenes():
    i=0
    # Abrir el archivo CSV
    with open('orders.csv', 'r') as archivo_csv:
        # Crear un objeto csv.reader para leer el archivo
        lector_csv = csv.reader(archivo_csv)
        # Iterar sobre las filas del archivo
        for fila in lector_csv:
            # Procesar la fila según sea necesario
            valor = str(fila)
            valor = valor.replace("[", "")
            valor = valor.replace("]", "")
            valor = valor.replace("'", "")
            valor = valor.replace("'", "")
            valor = valor.replace("P", "")
            if valor.startswith("Order "):
                i+=1
                print(valor)
    print("Total de ordenes", i)
