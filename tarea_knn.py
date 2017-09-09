# Copyright 2017 por Gonzalo Mardones (cualquier uso debe ser informado a)
# gonzalo-a@hotmail.com
# Ejecucion: python tarea_knn.py (debe contar con archivo 'iris_dataset.csv')

import csv            # incluye funcionalidades para el trabajo con archivos formato csv
import random		  # incluye funcionalidades de random
import math			  # incluye funcionalidades matematicas
import operator		  # Modulo de funcionalidades varias python
import numpy as np    # Numpy: manejo de arreglos
import matplotlib.pyplot as plt # Modulo de graficos

''' 
cargarSegmentos: funcion responsable de cargar los segmentos en lista_segmentos
cantidad_segmentos: recibe la cantidad de segmentos solicitados
dataset: almacena los datos leidos desde formato csv
lista_segmento: almacena la lista de los segmentos generados
'''
def cargarSegmentos(cantidad_segmentos,dataset, lista_segmento):
	with open('iris_dataset.csv', 'rb') as archivo_csv: # lee el directorio CSV
	    linea = csv.reader(archivo_csv) # lee cada linea del archivo CSV
	    dataset = list(linea) # dataset posee una lista con cada elemento del dataset 
	    random.shuffle(dataset) # lista aleatoria, suffle mezcla una lista generando una lista
	    x = len(dataset)/cantidad_segmentos #obtengo el tamanio de instancias para cada segmento, resultado = 15 eltos por segmento
	    for i in range(cantidad_segmentos):
	    	lista_segmento.append(dataset[(i*x): (i*x + x)]) #almaceno cada sub-segmentos en una unica lista
 
'''
distancia_eucliana: funcion responsable de retornar la distancia 
de la instancia de test y de la instancia de entrenamiento.
'''
def distanciaEucliana(punto_1, punto_2, largo):
	distance = 0
	for x in range(largo):
		distance += pow( float(punto_1[x]) - float(punto_2[x]), 2)
	return math.sqrt(distance)

'''
Vecinos: funcion responsable de retornar la lisata de vecinos para cada instancia de testeo evaluada
en la lista de entrenamiento.
training: lista de entrenamiento
instancia_test: instancia de test
k: cuantos vecinos se busca evaluar
'''
def Vecinos(training, instancia_test, k):
	distancia = [] 
	largo = len(instancia_test)-1 # largo de la instancia test
	for x in range(len(training)): # para elemento de la lista de entremamiento, evaluo la distancia con test
		dist = distanciaEucliana(instancia_test, training[x], largo)
		distancia.append((training[x], dist)) # se aprega la tupla, instancia de entrenamiento y distancia generada
	distancia.sort(key=operator.itemgetter(1)) # se ordena por el elemento 1er elemento de la tupla, osea, por distancia
	vecinos = []
	for x in range(k):
		vecinos.append(distancia[x][0]) #almaceno las distancias de los vecinos k buscados
	return vecinos
 
'''
getResponse: funcion responsable de obtener el tipo de iris que obtuvo mas votos 
'''
def obtenerRespuesta(vecinos):
	classVotes = {} # Gerera tupla iris - vecinos, ej:  {'Iris-virginica': 1}
	for x in range(len(vecinos)):

		response = vecinos[x][-1] #se obtiene el tipo de iris: ej: response:  Iris-virginica 
		if response in classVotes: 
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True) # ordena la tupla sortedVotes:  [('Iris-setosa', 2)]
	return sortedVotes[0][0] #retorna tipo de iris, ej: sortedVotes[0][0]:  Iris-setosa
 
'''
obtenerExactitud: funcion responsable de obtener la exactitud de la lista de predicciones
y la lista de test, ej:
prediccion: ['Iris-setosa', 'Iris-setosa',..., 'Iris-setosa']
testSet: [['5.0', '3.3', '1.4', '0.2', 'Iris-setosa'], ['4.4', '3.0', '1.3', '0.2', 'Iris-setosa'],...,['4.6', '3.1', '1.5', '0.2', 'Iris-setosa']]
retornando la exactitud de la prediccion en la lista Test
'''
def obtenerExactitud(test, prediccion):
	correcto = 0
	for x in range(len(test)):
		if test[x][-1] == prediccion[x]:
			correcto += 1
	return (correcto/float(len(test))) * 100.0

def main():

	fig = plt.figure() # inicio de modulo grafico.
	fig.canvas.set_window_title('Machine Learning/ Tarea 1 - KNN. Autor: Gonzalo Mardones')
	plt.xlim(1, 10) # Rango eje X
	plt.ylim(92, 102) # Rango eje Y
	eje_x = [] 
	eje_y = []
	plt.ylabel('Exactitud Promedio') # titulo eje y
	plt.xlabel('K valores') # titulo eje x
	
	#preparacion de variables
	training = [] 
	lista_segmento = [] #almacena la lista de segmentos (10 segmentos)
	dataset = [] 
	cantidad_segmentos = 10 
	ks = 10 # para k = 10

	cargarSegmentos(cantidad_segmentos, dataset, lista_segmento) 

	# Generar prediccion
	for k in range(1,(ks+1)): 
		exactitud = 0 # para cada k calculo su exactitud
		for x in range(len(lista_segmento)): # para cada uno de los segmentos existentes
			prediccion=[] # almaceno las predicciones generadas
			training = np.delete(lista_segmento,x,axis=0) # elimino un segmento intercalado para formar la lista de entremamiento
			training = np.array(np.concatenate(training)) # uno las listas y obtengo una unica lista de entrenamiento
			for y in range(len(lista_segmento[x])): # para cada uno de los segmentos
				vecinos = Vecinos(training, np.array(lista_segmento[x][y]), k) # evaluo para la lista de entremaniento, de test y k
				resultado = obtenerRespuesta(vecinos) # para cada uno de los vecinos obtengo la prediccion
				prediccion.append(resultado) 
			exactitud += obtenerExactitud(lista_segmento[x], prediccion) # se obtiene la exactitud
		exactitud/=len(lista_segmento) # para cada k se obtiene la exactitud promedio
		print 'k = '+ str(k)+', Exactitud: ' + str(exactitud) + '%'
		eje_x.append(k) # se agrega valor k para graficar
		eje_y.append(exactitud) # se agrega valor de exactitud para graficar
	
	#a =  np.array(eje_y)
	#print "a: ",a.mean()

	# plot grafico 
	plt.plot(eje_x,eje_y,'g--d')
	max_x = eje_x[eje_y.index(np.max(eje_y))]
	max_y = np.max(eje_y)

	min_x = eje_x[eje_y.index(np.min(eje_y))]
	min_y = np.min(eje_y)
	valores = [[max_x, min_x], [max_y, min_y,]]
	etiquetas_fil = ('K', 'Ext')
	etiquetas_col = (u'Maximo', u'Minimo')

	plt.table(cellText=valores, rowLabels=etiquetas_fil,
	colLabels = etiquetas_col, colWidths = [0.3]*len(eje_y), loc='upper center')
	plt.show()

#Funcion de iniciacion - main()
main()