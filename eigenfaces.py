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
#interactive sólo se necesita cuando se está corriendo desde el shell, comentar si se usa Spyder
from matplotlib import interactive
interactive(True)

def pcag(data):
    mu = sp.mean(data,axis=0)
    x = data - mu
    gram = x @ x.T
    latent,coeff = sp.linalg.eigh(gram)
    #latent = latent.real necesario si uso eig en vez de eigh, los autovalores tienen una parte imaginaria que no necesito
    #coeff = coeff.real
    #revisar esta división, es más complicado en scipy
    #coeff = (x.T @ coeff)/sp.sqrt(sp.absolute(sp.diag(latent))) Esta era la idea básica
    coeff = sp.linalg.solve(sp.sqrt(sp.absolute(sp.diag(latent))).T,coeff.T @ x).T
    latent = latent/(x.shape[0]-1)
    index = sp.argsort(-latent)
    latent = latent[index]
    coeff = coeff[:,index]
    explained = 100 * (latent/sp.sum(latent))
    score = x @ coeff
    return coeff,score,latent,explained,mu

#wdir = r"C:\Users\Federico\Documents\Python\Eigenfaces test\\"
wdir = os.getcwd() + "\\"
S = ("kaknig.","phughe.","vstros.")
filespergroup = 10
filestoread = len(S) * filespergroup
T = sp.zeros((filestoread,))
IM = sp.zeros((filestoread, 180 * 200 * 3))
for ngroup in range(len(S)):
    for nfile in range(filespergroup):
        img = io.imread(wdir + S[ngroup] + str(nfile + 1) + ".jpg")
        IM[ngroup * filespergroup + nfile,:] = sp.ravel(img)
        T[ngroup * filespergroup + nfile] = ngroup
        

coeff,score,latent,explained,mu = pcag(IM)

print(coeff,score,latent,explained,mu,sep="\n\n")

ms =  [ 'o' , '^' , 's' , 'v' , '>' , 'o' , '<' ] 
mfc = [  'b'  , 'y' , 'w' , 'g' , 'c' , 'k' , 'm' ]

fig = plt.figure("PCAfigure") #crea la figura y dibuja sobre ella desde ahora

plt.title('PCA - Clasificador lineal')

xaxislabel = "Primera Componente Principal %2.2f %%" % explained[0]
yaxislabel = "Segunda Componente Principal %2.2f %%" % explained[1]

plt.xlabel(xaxislabel)
plt.ylabel(yaxislabel)

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

N = 4 #número de dimensiones para clasificación

#entrena el clasificador
clf = CLF()
clf.fit(score[:,0:N], T)

#descomentar todo lo que sigue si se desea probar clasificar uno de los rostros desde el shell;
#en Spyder las variables no desaparecen al correr classifyface.py por lo que puedo correr ese script en vez de las siguientes líneas

#img = S[0] + "16.jpg"   

#A = io.imread(img)

#D = sp.ravel(A)

#E = (D - mu) @ coeff

#predictedclass = clf.predict(E[0:N].reshape((1,-1)))
    
#plt.plot(E[0],E[1], 'x' , markersize = 9 ,linewidth = 1, markerfacecolor = 'k' , markeredgecolor= 'k'      )

#print(predictedclass)

#plt.text( E[0],E[1], "   Grupo: " + S[int(predictedclass)] ,fontsize =  8 )
