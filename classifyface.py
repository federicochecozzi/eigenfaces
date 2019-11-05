# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:19:48 2019

@author: Federico
"""

#Este script está pensado para correrse en Spyder, en donde las variables
#permanecen en memoria; si no es el caso correr eigenfaces.py con el final del
#código sin comentar

img = S[0] + "20.jpg"   

A = io.imread(img)

D = sp.ravel(A)

E = (D - mu) @ coeff

predictedclass = clf.predict(E[0:N].reshape((1,-1)))

plt.figure("PCAfigure")
    
plt.plot(E[0],E[1], 'x' , markersize = 9 ,linewidth = 1, markerfacecolor = 'k' , markeredgecolor= 'k'      )

print(predictedclass)

plt.text( E[0],E[1], "   Grupo: " + S[int(predictedclass)] ,fontsize =  8 )