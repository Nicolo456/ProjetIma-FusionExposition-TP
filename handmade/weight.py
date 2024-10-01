import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Damien/ALL/Codage/Python/Filtres')
import filtres as fil
from math import *

#scikit-image contient beaucoup de fonctions déjà faites pour le traitement d'image


im = 'images/test.png'
im1 = 'papier/boat_low.png'
im2 = 'papier/boat_mid.png'
im3 = 'papier/boat_high.png'

###Calcul du contraste

def contraste(im, affichage=False):
    im_gris = fil.niveau_de_gris(im)          #conversion en niveau de gris
    im_lapl = fil.filtre(fil.laplacien_3,im_gris,False)       #application du filtre laplacien
    C = abs(im_lapl)                    #calcul du contraste
    if affichage:
        fil.afficher_1_image(im)
        fil.afficher_1_image(im_gris)
        fil.afficher_1_image(C)
    return C

###Calcul de la saturation 

def saturation(imt, affichage=False):
    if type(imt) == str:
        imt = plt.imread(imt)
    N, M = len(imt), len(imt[0])
    S = [[0 for i in range(M)] for j in range(N)]       #Saturation
    for i in range(N):
        for j in range(M):
            r, g, b = imt[i][j][0], imt[i][j][1], imt[i][j][2]
            moyenne = (r + g + b)/3
            variance = ((r-moyenne)**2 + (g-moyenne)**2 + (b-moyenne)**2)/3
            S[i][j] = sqrt(variance)
    if affichage:
        fil.afficher_1_image(S)
    return S

###Calcul de l'indice d'exposition --> on aurait pu le calculer en même temps que la saturation mais je le fais après pour plus de clarté

def exposition(imt, affichage=False):
    if type(imt) == str:
        imt = plt.imread(imt)
    N, M = len(imt), len(imt[0])
    sigma = 0.2
    E = [[0 for i in range(M)] for j in range(N)]       #indice d'exposition
    for i in range(N):
        for j in range(M):
            r, g, b = imt[i][j][0], imt[i][j][1], imt[i][j][2]
            e_r = exp(-((r-0.5)**2)/(2*sigma**2))
            e_g = exp(-((g-0.5)**2)/(2*sigma**2))
            e_b = exp(-((b-0.5)**2)/(2*sigma**2))
            E[i][j] = e_r * e_g * e_g
    if affichage:
        fil.afficher_1_image(E)
    return E

###Calcul de la carte des poids

def weightmap(im, w_c=1, w_s=1, w_e=1):
    imt = plt.imread(im)
    N,M = len(imt), len(imt[0])
    C = contraste(im)
    S = saturation(im)
    E = exposition(im)
    W = [[C[i][j]**w_c * S[i][j]**w_s * E[i][j]**w_e + 10**(-5) for j in range(M)] for i in range(N)]
    fil.afficher_1_image(W)
    return W

###Normalisation des weightmap:

def normalisation(weight_list):
    l = len(weight_list)
    N, M = len(weight_list[0]), len(weight_list[0][0])
    for i in range(N):
        for j in range(M):
            somme = 0
            for k in range(l):
                somme += weight_list[k][i][j]
            for k in range(l):
                weight_list[k][i][j] =  weight_list[k][i][j]/somme
    return weight_list


###Fusion des weightmap: 

def fusion(im_list):
    l = len(im_list)
    imt_list = [plt.imread(im_list[i]) for i in range(l)]
    N, M = len(imt_list[0]), len(imt_list[0][0])
    weight_list = [weightmap(im_list[i]) for i in range(l)]
    weight_norm = normalisation(weight_list)
    im_fus = [[0 for j in range(M)] for i in range(N)]
    for i in range(N):
        for j in range(M):
            pixel = 0
            for k in range(l):
                pixel += weight_norm[k][i][j]*imt_list[k][i][j]
            im_fus[i][j] = pixel
    fil.afficher_1_image(im_fus)
    return im_fus

fusion([im1,im2,im3])
