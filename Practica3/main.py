"""
Práctica 3 OpenCV.
Alberto Armijo Ruiz.
4º GII
"""

import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from auxFunc import *
from os import listdir
from os.path import join,isfile
import pickle
from numpy.linalg import norm

def readImage(imagePath):
    image = cv2.imread(imagePath,0)
    image = np.float32(image)
    image = cv2.normalize(image,image,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
    return image

def mostrarMultiplesImagenes(imagenes, titulos):
    # Calculamos el número de filas y columnas en las que tenemos que dividir
    # nuestro recuadro. Si es entero, en n*n cuadrados, sino en (n+1)^2 cuadrados.s
    n = math.sqrt(len(imagenes))
    fil = col = n

    if (n - int(n) > 0):
        fil = col = n + 1

    # pintamos las imágenes.
    for i in range(len(imagenes)):
        img = imagenes[i]
        plt.subplot(fil, col, i + 1)
        plt.title(titulos[i])
        plt.imshow(img)

    plt.show()

def calculateDescriptors(image,mask=None):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp,descriptors = sift.detectAndCompute(image,mask)

    sol=[kp,descriptors]
    return sol

# Función para calcular correspondencisa por 2nn.
def correspondencias2NN(kp1,desc1,kp2,desc2):
    # declaramos el matcher.
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2,crossCheck=False)
    # obtenemos correspondencias.
    correspondencias = matcher.knnMatch(desc1,desc2,k=2)

    bestCorr = []
    threshold = 0.65

    #obetenemos aquellas correspondencias que son buenas.
    for dch, izq in correspondencias:
        if dch.distance < threshold*izq.distance:
            bestCorr.append([dch])

    # Ordenamos las correspondencias y nos quedamos con las 50 mejores.
    bestCorr = sorted(bestCorr,key=lambda dist:dist[0].distance, reverse=False)[:50]

    return bestCorr

#-----------------------------------------------------------------------------------------------------------
# Ejercicio 1.
# Función para calcular una máscara.
def createMask(img,maskPoints):
    # Definimos una matriz del mismo tamaño que la imagen.
    mask = np.zeros(img.shape,dtype=np.uint8)

    # Transformamos los puntos en un array.
    ROI_points = np.array(maskPoints,dtype=np.int64)

    # Creamos un array con los puntos que hemos elegido.
    aprox =  cv2.approxPolyDP(ROI_points, 1.0,True)
    # Pasamos los puntos a la matriz que hemos creado anteriormente.
    cv2.fillConvexPoly(mask, aprox,color=(255,255,255))

    # Devolvemos la máscara.
    return mask

# Función para calcular las correspondencias de las imágenes, pero la primera de ello con una máscara.
def calculateCorrespondencesWithMask(img,img2,mask):
    # calculamos los descriptores.
    kp1, desc1 = calculateDescriptors(img, mask)
    kp2,desc2 = calculateDescriptors(img2)

    # Calculamos las correpondencias.
    corr = correspondencias2NN(kp1,desc1,kp2,desc2)

    # Dibujamos las correspondencias.
    matchesDrawn = cv2.drawMatchesKnn(img1=img,keypoints1=kp1,img2=img2,keypoints2=kp2,matches1to2=corr,outImg=None)
    matchesDrawn = cv2.cvtColor(matchesDrawn,cv2.COLOR_BGR2RGB)
    plt.imshow(matchesDrawn)
    plt.show()

#-----------------------------------------------------------------------------------------------------------
# Ejercicio 2.
# Función para calcular los 20 descriptores más cercanos y la varianza de los 10 más cercanos.
def calculateNearestDescriptors(descriptors,word):

    # Calculamos las distancias entre la palabra y el descriptor.
    distancesDescWord = [
        [ind,cv2.norm(src1=word,src2=desc, normType=cv2.NORM_L2)] for ind,desc in descriptors.items()
    ]

    # Ordenamos por menor distancia y nos quedamos con los 20 primeros.
    distancesDescWord = sorted(distancesDescWord, key=lambda d: d[1])[:20]

    # Calculamos la variaza de las mejores 10.
    selected = np.array(distancesDescWord[:10])
    selected = selected[:,1]
    # Calculamos la varianza.
    variance = np.var(selected)

    distancesDescWord = dict(distancesDescWord)

    # Devolvemos la varianza y los 20 descriptores más cercanos.
    return [variance,distancesDescWord]

# Función para calcular las 20 parches más cercanos.
def calculateNearestPatches(patches,bestDescriptors):
    # Con los índices de los descriptores obtenemos los parches que son más cercanos.
    nearestPatches = dict(
        [[descID,patches[descID]] for descID in bestDescriptors.keys()]
    )

    return nearestPatches

# Función para obtener el subconjunto de descriptores que pertenecen a un cluster.
# dado por una palabra visual.
def obtainDescriptorsPerWord(word_index,descriptors,labels):
    # Obtenemos los índices de los descriptores.
    desc_index = [
        ind for ind in range(len(labels)) if labels[ind] == word_index
    ]

    # Guardamos el subconjunto en un diccionario.
    word_descriptors = dict(
        [ind, descriptors[ind]] for ind in desc_index
    )

    # Devolvemos el subconjunto.
    return word_descriptors

# Función para obtener la varianza y los 20 mejores parches de cada palabra.
def calculateVarianceAndBestPatches(vocabulary,descriptors,labels,patches):
    # Creamos un diccionario donde guardaremos variaza y 20 mejores parches por cada palabra.
    varAndPatches = list()
    bestClusters = list()

    for i in range(len(vocabulary)):
        print("calculating for word: ",i)
        selected_word = i
        descriptors_subset = obtainDescriptorsPerWord(word_index=selected_word, descriptors=descriptors, labels=labels)
        var, best_desc = calculateNearestDescriptors(descriptors=descriptors_subset, word=vocabulary[selected_word])
        best_patches = calculateNearestPatches(patches, best_desc)

        # Anadimos elemento a nuestro diccionario.
        varAndPatches.append([var,best_patches])
        bestClusters.append([selected_word,var])

    # Ordenamos por menor varianza.
    bestClusters = sorted(bestClusters,key=lambda x:x[1])

    return [bestClusters,varAndPatches]

# Función para obtener la varianza y los 20 mejores parches de cada palabra.
def calculateVarianceAndBestPatchesWord(word_index,vocabulary,descriptors,labels,patches):
    # calculamos los parches y la varianza de la palabra.
    descriptors_subset = obtainDescriptorsPerWord(word_index=word_index, descriptors=descriptors, labels=labels)
    var, best_desc = calculateNearestDescriptors(descriptors=descriptors_subset, word=vocabulary[word_index])
    best_patches = calculateNearestPatches(patches, best_desc)

    # devolemos la varianza y los parches.s
    return [var,best_patches]

#-----------------------------------------------------------------------------------------------------------
# Ejercicio 3.

# Funciones para guardar y leer el índice invertido.
# Ver https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Función para calcular las palabras más votadas de una imagen.
def calculateVotes(img,dictionary,max_words=200):

    #Calculamos os decriptores de la imagen.
    kp, desc = calculateDescriptors(image=img)

    # Normalizamos los descriptores de la imagen.
    norm_desc = desc
    cv2.normalize(src=desc,dst=norm_desc,norm_type=cv2.NORM_L2)
    cv2.normalize(src=dictionary,dst=dictionary,norm_type=cv2.NORM_L2)

    # calculamos de semejanza entre las palabras y los descriptores.
    vote_matrix = np.dot(dictionary,norm_desc.T)


    # Calculamos las palabras que han sido más votadas para la imagen.
    words_of_image = np.zeros(shape=vote_matrix.shape[0], dtype=np.int)
    for col in range(vote_matrix.shape[1]):
        # Obtenemos la matriz.
        desc_column = vote_matrix[:, col]
        # Obtenemos la palabra a la que tiene la distancia menor.
        min_index = np.argmax(desc_column)
        # Aumentamos en uno el número de votos.
        words_of_image[min_index] += 1


    # Guardamos palabras y número de votos.
    real_words = [[index, words_of_image[index]] for index in range(len(words_of_image))]
    real_words = dict(real_words)
    # Nos quedamos con los 20 más votados.
    #real_words = sorted(real_words, key=lambda elem: elem[1], reverse=True)[:max_words]
    real_words = dict( [index,votes] for index, votes in real_words.items() if votes > 0 )


    return [real_words,words_of_image]

# Función para devolver todos los nombres de imágenes de un directorio.
def listImagesNames(path):
    listaNombres = [imagen for imagen in listdir(path) if isfile(join(path,imagen)) and '.pkl' not in imagen ]

    return listaNombres

# Función para crear el fichero de índice invertido.
def createInvertedIndex(dictionary,path='./imágenes'):
    nombreImagenes = listImagesNames(path)
    invertedIndex = dict([index,[]] for index in range(dictionary.shape[0]))
    BagOfWords = dict([name,[]] for name in nombreImagenes)

    for name in nombreImagenes:
        print("calculando votos para:", name)
        imagen = cv2.imread(join(path,name))
        votes, bag_of_words = calculateVotes(img=imagen,dictionary=dictionary)

        BagOfWords[name] = bag_of_words

        for index,vot in votes.items():
            invertedIndex[index].append(name)

    return [invertedIndex, BagOfWords]

# Función para obtener imágenes semejantes mediante las palabras de una imagen.
def obtainImagesFromQueryImage(dictionary,query_image, inverted_index):
    # Calculamos los votos de la imágenes.
    words, bag = calculateVotes(img=query_image,dictionary=dictionary)

    # Obtenemos las imágenes que también tienen esas palabras.
    images = set()
    for palabra, vot in words.items():
        for name in  inverted_index[palabra]:
            images.add(name)

    return images

# Función para obtener las palabras de una imagen a través del fichero invertido.
def obtainWordsFromInvertedFile(filename,inverted_file):
    # Obtenemos una lista con las palabras que contienen esa imagen.
    words_of_image = [word for word, files in inverted_file.items() if filename in files]

    return words_of_image

# Función para calcular los descriptores de las imágenes.
def calculateDescriptorsFromWords(words,dictionary):
    desc = [dictionary[word] for word in words]

    desc = np.array(desc)

    return desc

# Función para normalizar los datos de un vector unidimensional.
def calculateNormalizedVector(x):
    max = np.max(x)
    min = np.min(x)

    z = np.array([(x_i-min)/(max-min) for x_i in x ])

    return z

# Función para calcular aquellas imágenes que son más semejantes una imagen.
def calculateSimilarImages(img,inverted_file,dictionary, bagOfImages):
    # Calculamos las imágenes que tienen los mismas palabras que la imagen.
    imgs = obtainImagesFromQueryImage(dictionary=dictionary,query_image=img,inverted_index=inverted_file)

    # calculamos la bolsa de palabras de las imagenes que tienen
    # palabras parecidas con las palabras de la imagen.
    bag_of_words_per_image = dict([name,bagOfImages[name]] for name in imgs )
    # Calculamos la bolsa de palabras de nuestra imagen.
    vot, queryBoW = calculateVotes(img=img,dictionary=dictionary)


    # Para cada imagen de bag_of_words_per_image calculamos la semejanza con nuestra imagen.

    semejanzaImgs = [ [name, ( np.dot(queryBoW,imageBow.T) ) / (norm(queryBoW)*norm(imageBow))]
                            for name,imageBow in bag_of_words_per_image.items()]

    semejanzaImgs = sorted(semejanzaImgs, key=lambda par: par[1], reverse=True)[:5]

    semejanzaImgs = dict(semejanzaImgs)

    return semejanzaImgs

#-----------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------
#Ejercicio 1.
#------------------------------------------------------------------------
img = cv2.imread("imágenes/128.png")
img2 = cv2.imread("imágenes/130.png")
rows,cols = img.shape[:2]
print(cols,rows)
# Diferentes máscaras para el ejercicio 1.
mask1 = [(0,rows/4),(cols/4,0),(cols/2,rows/4),(cols/4,rows/2)]
mask2 = [(cols/2,rows/4),(3*cols/4,0),(cols,rows/4),(3*cols/4,rows/2)]
mask3 = [(0,3*rows/4),(cols/4,rows/2),(cols/2,3*rows/4),(cols/4,rows)]
mask4 = [(cols/2,3*rows/4),(3*cols/4,rows/2),(cols,3*rows/4),(3*cols/4,rows)]
mask5 = [(cols/4,rows/2),(cols/2,rows/4),(3*cols/4,rows/2),(cols/2,3*rows/4)]
masks = [mask1,mask2,mask3,mask4,mask5]
mask = createMask(img,mask2)

mask = np.array(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY))

calculateCorrespondencesWithMask(img=img, img2=img2,mask=mask)

img = cv2.imread("./imágenes/229.png")
img2 = cv2.imread("./imágenes/232.png")

mask = createMask(img,mask1)

mask = np.array(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY))

calculateCorrespondencesWithMask(img=img,img2=img2,mask=mask)

#------------------------------------------------------------------------
# Ejercicio 2.
#------------------------------------------------------------------------
# Para la documentación: https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
#Leemos el archivo de descriptores y parches.
descriptors, patches = loadAux("./imágenes/descriptorsAndpatches.pkl",flagPatches=True)
accu, labels, vocabulary = loadDictionary("./imágenes/kmeanscenters5000.pkl")

#bestClusters, varsAndPatches = calculateVarianceAndBestPatches(vocabulary,descriptors,labels,patches)

var,best_patches = calculateVarianceAndBestPatchesWord(word_index=931,vocabulary=vocabulary,descriptors=descriptors
                                                   ,labels=labels,patches=patches)

imags = [imag for id,imag in best_patches.items()]
tit = [str(i) for i in range(len(best_patches))]
mostrarMultiplesImagenes(imagenes=imags,titulos=tit)

var,best_patches = calculateVarianceAndBestPatchesWord(word_index=899,vocabulary=vocabulary,descriptors=descriptors
                                                   ,labels=labels,patches=patches)

imags = [imag for id,imag in best_patches.items()]
tit = [str(i) for i in range(len(best_patches))]
mostrarMultiplesImagenes(imagenes=imags,titulos=tit)

var,best_patches = calculateVarianceAndBestPatchesWord(word_index=4054,vocabulary=vocabulary,descriptors=descriptors
                                                   ,labels=labels,patches=patches)

imags = [imag for id,imag in best_patches.items()]
tit = [str(i) for i in range(len(best_patches))]
mostrarMultiplesImagenes(imagenes=imags,titulos=tit)

#------------------------------------------------------------------------
# Ejercicio 3.
#------------------------------------------------------------------------
#El tercer array son las palabras del diccionario.
dictionary = loadDictionary("./imágenes/kmeanscenters5000.pkl")
words = dictionary[2]

img = cv2.imread("imágenes/0.png")
aux = cv2.cvtColor(src=img,code=cv2.COLOR_BGR2RGB)

index, bagOfWords = createInvertedIndex(dictionary=words)

sem = calculateSimilarImages(img,index,words,bagOfWords)

imags = [cv2.cvtColor(src=cv2.imread('./imágenes/'+name),code=cv2.COLOR_BGR2RGB) for name,sim in sem.items()]
imags.append(aux)
tit = [str(i) for i in range(len(sem))]
tit.append('Original')
mostrarMultiplesImagenes(imagenes=imags,titulos=tit)

img2 = cv2.imread('./imágenes/15.png')
aux = cv2.cvtColor(src=img2,code=cv2.COLOR_BGR2RGB)


sem = calculateSimilarImages(img=img2, inverted_file=index,dictionary=words,bagOfImages=bagOfWords)

imags = [cv2.cvtColor(src=cv2.imread('./imágenes/'+name),code=cv2.COLOR_BGR2RGB) for name,sim in sem.items()]
imags.append(aux)
tit = [str(i) for i in range(len(sem))]
tit.append('Original')
mostrarMultiplesImagenes(imagenes=imags,titulos=tit)

img2 = cv2.imread('./imágenes/158.png')
aux = cv2.cvtColor(src=img2,code=cv2.COLOR_BGR2RGB)


sem = calculateSimilarImages(img=img2, inverted_file=index,dictionary=words,bagOfImages=bagOfWords)

imags = [cv2.cvtColor(src=cv2.imread('./imágenes/'+name),code=cv2.COLOR_BGR2RGB) for name,sim in sem.items()]
imags.append(aux)
tit = [str(i) for i in range(len(sem))]
tit.append('Original')
mostrarMultiplesImagenes(imagenes=imags,titulos=tit)
