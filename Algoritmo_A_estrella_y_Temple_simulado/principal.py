#imports
from cmd import Cmd
from interfaz_grafica import *
import os
from a_estrella.a_estrella import *
import interprete_de_ordenes
from a_estrella import a_estrella
from temple import temple
from layout import Layout


# Definir layout inicial
tam_estanteria = (6, 2)
tam_pasillo = 1
filas = 24
columnas = 12
tam_borde = 2
bahia_carga = [2, 0]
end = (7, 6)

class Consola(Cmd):

    def do_test (self, arg):    # ALMACEN
        """test"""

    def do_1 (self, arg):    # ALMACEN
        """Visualizacion del almacen"""
        interfaz(matriz, bahia_carga,ruta=None, almacen=None)

    def do_2 (self, arg):    # Inicio y ordenes
        """ver ordenes"""
        interprete_de_ordenes.ver_Ordenes()
    
    def do_3 (self, arg):    # ALGORITMO A*
        
        """El algoritmo A* calcula el camino mas optimo entre 2 puntos"""
        
        interfaz(matriz, bahia_carga)

        orders = interprete_de_ordenes.leer_ordenenes()
        costosFinal,almacen = a_estrella_iniciar(matriz.layout_matrix, bahia_carga, end, orders)

        """
        print(">>><<<")
        for a in costosFinal:
            print(a)
        print(">>><<<")
        """
        print("Inciciar temple?")
        if input("Y/N")=="Y":
            mejor = temple.iniciar_temple(costosFinal)
            interfaz(matriz, bahia_carga, mejor, almacen)
    
    def do_7 (self, arg): # SALIR
        """Salir del programa"""
        print("Saliendo del programa...")
        return True

    def precmd(self, line):
        print('Ejecutando comando: {}'.format(line))
        return line
    
    def postcmd(self, stop, line):
        print('Comando ejecutado.')
        return stop

if __name__=='__main__':    # INICIALIZACION DEL PROGRAMA
    
    os.system ('cls')
    
    matriz = Layout(tam_estanteria, tam_pasillo, filas, columnas, tam_borde, bahia_carga)
    menu = Consola()
    menu.prompt = ' >> '
    menu.doc_header = 'Indice de comandos:'
    text="\n'INTELIGENCIA ARTIFICIAL 2'  Grupo 6:\n\n" + \
        " Indice de comandos:\n\n"+\
        "   - Ver escenario (1)\n"+\
        "   - Ver ordenes (2)\n"+\
        "   - Algoritmo A* (3)\n"+\
        "   - Exit (7)\n\n"
    menu.cmdloop(text)      
