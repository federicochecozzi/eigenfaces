# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:14:34 2019

@author: Federico
"""

import scipy as sp
#from scipy import linalg
#import skimage
from skimage import io
import os
import matplotlib.pyplot as plt
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as CLF
from sklearn.linear_model import SGDClassifier as CLF
#from sklearn.svm import SVC as CLF

#IMPORTANTE: interactive sólo se necesita cuando se está corriendo desde el
#Shell, comentar si se usa Spyder, en Spyder usar el modo auto en vez de inline
#para ver figuras, sino las imprime en consola
from matplotlib import interactive
interactive(True)

def pcag(data):
    #implementación del algoritmo de PCA para grandes cantidades de datos
    #básicamente reinventa la rueda porque scikit parece tener un algoritmo
    #de PCA truncado que hace irrelevante este trabajo, pero es un buen
    #ejercicio de numpy
    
    #data: array cuyas filas corresponden a observaciones y columnas a variables
    #coeff: base ortonormal del espacio de componentes principales,
    #los vectores son componentes principales
    #score: proyección de obs. en data en el espacio de componentes principales
    #latent: autovalor/varianza asociada a cada uno de los vectores en coeff
    #explained: porcentaje de la varianza total explicado por cada componente
    #mu: vector con todos los valores medios de las variables de data
    
    mu = sp.mean(data,axis=0) #calcula el valor medio de todas las variables
    x = data - mu #resta fila a fila el valor medio, utilizando BROADCASTING
    gram = x @ x.T #calculo la matriz de Gram, cov(data) es mucho menos práctico
    latent,coeff = sp.linalg.eigh(gram) #los autovalores de cov(data) son los 
    #mismos que los de gram(x)/(N-1), los autovectores de cov(data) se obtienen 
    #premultiplicando los autovectores de gram(x) por x y dividiendo por la raíz
    #cuadrada de los autovalores; se ve más bonita la división en pcag.m
    #explicación: https://www.mii.lt/zilinskas/uploads/visualization/lectures/lect14/comparison.pdf
    coeff = sp.linalg.solve(sp.sqrt(sp.absolute(sp.diag(latent))).T,coeff.T @ x).T
    latent = latent/(x.shape[0]-1)#escalamiento por 1/(N-1) 
    index = sp.argsort(-latent)#obtiene los índices para ordenar autovalores
    #en forma descendente, los mayores autovalores representan componentes más importantes
    latent = latent[index]
    coeff = coeff[:,index]
    explained = 100 * (latent/sp.sum(latent))#var comp / var tot *100
    score = x @ coeff#la multiplicación hace los productos escalares para proyectar
    #en el espacio de componentes principales
    return coeff,score,latent,explained,mu

#PRIMERA PARTE: Análisis de componente principal de un grupo de imágenes vector,
#el objetivo es obtener una matriz score con vectores con menos dimensiones
    
wdir = os.getcwd() + "\\"#trabajo en el directorio del código fuente e imágenes
S = ("kaknig.","phughe.","vstros.")#tupla con los strings típicos de nombres de archivo
filespergroup = 10#uso diez archivos por persona
filestoread = len(S) * filespergroup#por lo que leo 3 X 10 archivos
T = sp.zeros((filestoread,))
IM = sp.zeros((filestoread, 180 * 200 * 3))
for ngroup in range(len(S)):
    for nfile in range(filespergroup):
        #leo una imagen y la cargo en un array
        img = io.imread(wdir + S[ngroup] + str(nfile + 1) + ".jpg")
        #convierto el array en un vector fila y lo guardo en la tabla de datos
        IM[ngroup * filespergroup + nfile,:] = sp.ravel(img)
        #almaceno a qué grupo pertenece la imágen para entrenar el clasificador
        T[ngroup * filespergroup + nfile] = ngroup
        

coeff,score,latent,explained,mu = pcag(IM)#hago el análisis de componentes 
#principales sobre el array con las imágenes

print(coeff,score,latent,explained,mu,sep="\n\n")

ms =  [ 'o' , '^' , 's' , 'v' , '>' , 'o' , '<' ] 
mfc = [  'b'  , 'y' , 'w' , 'g' , 'c' , 'k' , 'm' ]

fig = plt.figure("PCAfigure") #crea la figura y dibuja sobre ella desde ahora

plt.title('PCA - Clasificador lineal')

xaxislabel = "Primera Componente Principal %2.2f %%" % explained[0]
yaxislabel = "Segunda Componente Principal %2.2f %%" % explained[1]

plt.xlabel(xaxislabel)
plt.ylabel(yaxislabel)

#imprimo las proyecciones de cada imágen tomando las dos primeras dimensiones de score,
#que corresponden a las componentes principales (de coeff) con mayor porcentaje explicado
#recordemos que la función pcag ordena de forma descendente
#cada grupo tiene su formato propio (ver listas ms y mfc)
for jj in range(len(S)):
    mask = T == jj
    plt.plot(score[mask,0],score[mask,1], ms[jj], markersize = 9 , linewidth = 1, markerfacecolor = mfc[jj] , markeredgecolor = 'k')

f = 1.3
left, right = plt.xlim()
bottom, top = plt.ylim()
plt.xlim(f * left, f * right)
plt.ylim(f * bottom, f * top)
plt.legend(S)
#plt.show()#necesario si no uso interactive arriba

#SEGUNDA PARTE: Clasificación

N = 4 #número de dimensiones en el espacio de componentes principales a usar

#entrena el clasificador
#grupo de entrenamiento: imágenes expresadas en las primeras N dimensiones
#del espacio de componentes principales
#la etiqueta de clases está en el array T
clf = CLF()
clf.fit(score[:,0:N], T)#genera el model del clasificador

#descomentar todo lo que sigue si se desea probar clasificar uno de los rostros 
#desde el shell; en Spyder las variables no desaparecen al correr classifyface.py
#por lo que puedo correr ese script en vez de las siguientes líneas

img = S[0] + "16.jpg"

A = io.imread(img)

D = sp.ravel(A)#convierto a vector fila

E = (D - mu) @ coeff#se proyecta el vector centrado respecto de la media al 
#espacio de componentes principales

predictedclass = clf.predict(E[0:N].reshape((1,-1)))#clasifico la imágen
    
plt.plot(E[0],E[1], 'x' , markersize = 9 ,linewidth = 1, markerfacecolor = 'k' , markeredgecolor= 'k'      )

print(predictedclass)

plt.text( E[0],E[1], "   Grupo: " + S[int(predictedclass)] ,fontsize =  8 )
