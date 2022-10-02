# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-hidden,-heading_collapsed,-run_control,-trusted
#     cell_metadata_json: true
#     notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version,
#       -jupytext.text_representation.format_version, -language_info.version, -language_info.codemirror_mode.version,
#       -language_info.codemirror_mode, -language_info.file_extension, -language_info.mimetype,
#       -toc
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#   nbhosting:
#     title: suite du TP simple avec des images
# ---

# %% [markdown]
# Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

# %%
from IPython.display import HTML
HTML(url="https://raw.githubusercontent.com/ue12-p22/python-numerique/main/notebooks/_static/style.html")



# %% [markdown]
# # suite du TP simple avec des images
#
# merci à Wikipedia et à stackoverflow
#
# **le but de ce TP n'est pas d'apprendre le traitement d'image  
# on se sert d'images pour égayer des exercices avec `numpy`  
# (et parce que quand on se trompe ça se voit)**

# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# %% [markdown] {"tags": ["framed_cell"]}
# **notions intervenant dans ce TP**
#
# sur les tableaux `numpy.ndarray`
#
# * `reshape()`, tests, masques booléens, *ufunc*, agrégation, opérations linéaires sur les `numpy.ndarray`
# * les autres notions utilisées sont rappelées (très succinctement)
#
# pour la lecture, l'écriture et l'affichage d'images
#
# * utilisez `plt.imread`, `plt.imshow`
# * utilisez `plt.show()` entre deux `plt.imshow()` dans la même cellule
#
# **note**
#
# * nous utilisons les fonctions de base sur les images de `pyplot` par souci de simplicité
# * nous ne signifions pas là du tout que ce sont les meilleures  
# par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
# alors que la fonction `save` de `PIL` le permet
#
# * vous êtes libres d'utiliser une autre librairie comme `opencv`  
#   si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte
#
# **n'oubliez pas d'utiliser le help en cas de problème.**

# %% [markdown]
# ## Création d'un patchwork

# %% [markdown]
# 1. Le fichier `rgb-codes.txt` contient une table de couleurs:
# ```
# AliceBlue 240 248 255
# AntiqueWhite 250 235 215
# Aqua 0 255 255
# .../...
# YellowGreen 154 205 50
# ```
# Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
# Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.
# <br>
#
# 1. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
# `'Red'`, `'Lime'`, `'Blue'`
# <br>
#
# 1. Faites une fonction `patchwork` qui  
#
#    * prend une liste de couleurs et la structure donnant le code des couleurs RGB
#    * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
#    * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
#    si besoin de compléter l'image mettez du blanc  
#    (`numpy.indices` peut être utilisé)
# <br>
# <br>   
# 1. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.
# <br>
#
# 1. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
# même chose pour des jaunes  
# <br>
#
# 1. Appliquez la fonction à toutes les couleurs du fichier  
# et sauver ce patchwork dans le fichier `patchwork.jpg` avec `plt.imsave`
# <br>
#
# 1. Relisez et affichez votre fichier  
#    attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels
#    
# vous devriez obtenir quelque chose comme ceci
# <img src="patchwork-all.jpg" width="200px">

# %%
# votre code

## 1 

import re
with open("rgb-codes.txt") as f:
    lines = f.readlines()
    colors = {}
    for line in lines: 
        elts = re.split(' |\n', line)
        colors[elts[0]] = (int(elts[1]), int(elts[2]), int(elts[3]))

print(colors)


## 2 

print(colors['Lime'], colors['Red'], colors['Blue'])


# %%
## 3

def patchwork(clist, colors):
    '''Retourne une image carré composée de pixels de couleur dans la liste clist ; taille côté image = len(clist)'''
    n = int(np.sqrt(len(colors.items())))
    pw_base = np.random.choice(clist, (n, n))
    pw = np.zeros((n, n, 3))
    for color in clist:
        rgb = colors[color]
        mask = (pw_base == color)
        pw[mask] = rgb
    pw = pw.astype(np.uint8)
    return pw
   

## 4  
    
clist = np.random.choice(list(colors.keys()), 25) # on choisit 25 couleurs aléatoirement dans le dictionnaire pour tester
pw = patchwork(clist, colors)

print(pw)

plt.imshow(pw)
plt.show()

# %%
## 5

white_list = []
for elt in colors.keys():
    if 'White' in elt:
        white_list.append(elt)
        
plt.imshow(patchwork(white_list, colors))
plt.show()

yellow_list = []
for elt in colors.keys():
    if 'Yellow' in elt:
        yellow_list.append(elt)
        
plt.imshow(patchwork(yellow_list, colors))
plt.show()

# %%
## 6 

pw = patchwork(list(colors.keys()), colors)
plt.imshow(pw)
plt.show()

plt.imsave("patchwork.jpg", pw)

# %%
## 7

pw_read = plt.imread("patchwork.jpg")
plt.imshow(pw_read)
plt.show()

# %% [markdown]
# ## Somme des valeurs RGB d'une image

# %% [markdown]
# 0. Lisez l'image `les-mines.jpg`
#
# 1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image  
#
# 2. Affichez l'image (pas terrible), son maximum et son type
#
# 3. Créez un nouveau tableau `numpy.ndarray` en sommant **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image
#
# 4. Affichez l'image, son maximum et son type
#
# 5. Pourquoi cette différence ? Utilisez le help `np.sum?`
#
# 6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
# (de la manière que vous préférez)
#
# 7. Remplacez dans l'image en niveaux de gris,   
# les valeurs >= à 127 par 255 et celles inférieures par 0  
# Affichez l'image avec une carte des couleurs des niveaux de gris  
# vous pouvez utilisez la fonction `numpy.where`
#
# 8. avec la fonction `numpy.unique`  
# regardez les valeurs différentes que vous avez dans votre image en noir et blanc

# %%
## 0

im = plt.imread("les-mines.jpg")
print(im.dtype)

## 1 

im1 = im[:, :, 0] + im[:, :, 1] + im[:, :, 2]

## 2

print(im1.max(), type(im1), im1.dtype)
plt.imshow(im1)
plt.show()

# %%
## 3

im2 = np.sum(im, axis = 2)

## 4 

print(im2.max(), type(im2), im2.dtype)
plt.imshow(im2)
plt.show()

# %%
## 5

# np.sum?

# %% [markdown]
# **5)** La documentation de la fonction précise : "Arithmetic is modular when using integer types, and no error is
# raised on overflow". Ainsi, le dtype de l'array `im2` est adapté de uint8 à la base (voir le dtype de `im` à la question 1) à uint32 pour éviter que la somme ne crée un overflow. Rester en uint8 comme pour l'image `im1` borne les valeurs de somme entre 0 et 255. Cela explique la différence entre les deux images.

# %%
## 6

im_gr = (im2 / im2.max() * 255).astype(np.uint8)
plt.imshow(im_gr, cmap = 'gray')
plt.show()

# %%
## 7

im_gr[im_gr < 127] = 0
im_gr[im_gr >= 127] = 255
im_gr.dtype

plt.imshow(im_gr, cmap = 'gray')

# %%
## 8

print(np.unique(im_gr))


# %% [markdown]
# ## Image en sépia

# %% [markdown]
# Pour passer en sépia les valeurs R, G et B d'un pixel  
# (encodées ici sur un entier non-signé 8 bits)  
#
# 1. on transforme les valeurs $R$, $G$ et $B$ par la transformation  
# $0.393\, R + 0.769\, G + 0.189\, B$  
# $0.349\, R + 0.686\, G + 0.168\, B$  
# $0.272\, R + 0.534\, G + 0.131\, B$  
# (attention les calculs doivent se faire en flottants pas en uint8  
# pour ne pas avoir, par exemple, 256 devenant 0)  
# 1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
# 1. naturellement l'image doit être ensuite remise dans un format correct  
# (uint8 ou float entre 0 et 1)

# %% [markdown]
# **Exercice**
#
# 1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
# la fonction `numpy.dot` doit être utilisée (si besoin, voir l'exemple ci-dessous) 
#
# 1. Passez votre patchwork de couleurs en sépia  
# Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso
# 2. Passez l'image `les-mines.jpg` en sépia   

# %%
## 1

def sepia(im):
    coeff = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    mat_coeff = np.ones((3, 3, 3))
    for k in range(3):
        mat_coeff[:, :, k] = coeff
    sepim = np.dot(im, mat_coeff)
    sepim = sepim / sepim.max()
    return sepim[:, :, :, 0]


# %%
## 2

pw = plt.imread("patchwork-all.jpg")
plt.imshow(sepia(pw))
plt.show()

# %%
## 3

im = plt.imread("les-mines.jpg")
plt.imshow(sepia(im))
plt.show()

# %% {"scrolled": true}
# INDICE:

# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

A.shape, B.shape, C.shape

# %% [markdown]
# ## Exemple de qualité de compression

# %% [markdown]
# 1. Importez la librairie `Image`de `PIL` (pillow)   
# (vous devez peut être installer PIL dans votre environnement)
# 1. Quelle est la taille du fichier 'les-mines.jpg' sur disque ?
# 1. Lisez le fichier 'les-mines.jpg' avec `Image.open` et avec `plt.imread`  
#
# 3. Vérifiez que les valeurs contenues dans les deux objets sont proches
#
# 4. Sauvez (toujours avec de nouveaux noms de fichiers)  
# l'image lue par `imread` avec `plt.imsave`  
# l'image lue par `Image.open` avec `save` et une `quality=100`  
# (`save` s'applique à l'objet créé par `Image.open`)
#
# 5. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
# Que constatez-vous ?
#
# 6. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence  

# %%
## 1 

from PIL import Image

# %%
## 2

import os
size = os.stat("les-mines.jpg")
print(f"La taille du fichier est : {size.st_size/1000} ko")



# %%
## 3

im1 = Image.open("les-mines.jpg")
im2 = plt.imread("les-mines.jpg")

# %%
## 4

# Pour vérifier que les valeurs contenues dans les array sont proches, on calcule la distance entre les deux tableaux. 
# En l'occurence, la distance est nulle, donc les valeurs sont très proches. 
print(np.linalg.norm(im1-im2))

# %%
## 5

im1.save('mines_PIL.jpg', quality = 100)
plt.imsave('mines_plt.jpg', im2)

# %%
## 6

size_PIL = os.stat("mines_PIL.jpg")
size_plt = os.stat("mines_plt.jpg")
print(f"taille du fichier avec PIL (save) : {size_PIL.st_size/1000} ko")
print(f"taille du fichier avec pyplot (imsave): {size_plt.st_size/1000} ko")
print("On remarque que l'image enregistrée avec PIL est plus 4 à 5 fois plus lourde que celle enregistrée avec pyplot. \nDans les deux cas la qualité est dégradée par rapport à l'image originale.")
print("D'ailleurs, on voyait bien à la question 7 de la première partie que le patchwork enregistré puis affiché était beaucoup plus terne que l'original.")

# %%
## 7

im1 = plt.imread("mines_PIL.jpg")
im2 = plt.imread("mines_plt.jpg")
plt.imshow(im1)
plt.show()
plt.imshow(im2)
plt.show()
print("Ici, les deux images semblent parfaitement identiques malgré leur différence de taille en mémoire dans le disque")
