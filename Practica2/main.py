# Práctica 2 VC.
# Alberto Armijo Ruiz.
# 4º GII.

import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import cv2.xfeatures2d

# Función para leer una imagen.
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
        plt.imshow(img, 'gray')

    plt.show()

def createGuassianPyramid(imagen,tipoBorde=0,imagenes=[],niveles=3):
    # Primero crearemos la imagen que vamos a devolver.
    height,width=imagen.shape
    total_height = int(height)
    total_width = int(1.5*width+1)

    # Imagen de la pirámide.
    gaussianPyramid=np.ones((total_height,total_width))

    # Pegamos la primera imagen dentro de la imagen original.
    gaussianPyramid[0:height,0:width]=imagen
    imagenes.append(copy.deepcopy(imagen))

    # Reducimos el tamaño de la imagen.
    size=(int(imagen.shape[1]/2),int(imagen.shape[0]/2) )
    new_image = cv2.pyrDown(imagen,dstsize=size)
    aux_h = 0


    # Aplicamos el mismo proceso por cada uno de los niveles siguientes de la pirámide.
    for i in range(0,niveles-1):
        imagenes.append(copy.deepcopy(new_image))
        gaussianPyramid[aux_h:aux_h+new_image.shape[0],width+1:width+1+new_image.shape[1]]=new_image
        aux_h = aux_h+new_image.shape[0] + 1

        size=(int(new_image.shape[1]/2),int(new_image.shape[0]/2) )
        new_image = cv2.pyrDown(new_image,dstsize=size)


    return gaussianPyramid

# Función para calcula la matriz de intensidades.
def calculateHarrysMatrix(src):
    mat = np.zeros(src.shape[:2],np.float32)
    rows,cols=src.shape[:2]
    # Recorremos filas y columnas, calculando por cada uno de los puntos,
    # la intensidad en cada uno de los puntos.
    for i in range(rows):
        for j in range(cols):
            lambda1=src[i,j,0]
            lambda2=src[i,j,1]
            mat[i,j]=lambda1*lambda2 - (0.04)*((lambda2+lambda1)**2)
            #mat[i,j]=lambda1*lambda2/(lambda1+lambda2)

    # Una vez hemos calculado los valores, devolvemos la matriz.
    return mat

# Funciones para la supresión de no máximos.

# Función para comprobar si el punto central es el máximo dentro de la ventana.
def comprobarMaximoCentral(matriz,tam_ventana,x,y):
    pos_central = math.floor(tam_ventana/2)
    ini_x = x-pos_central if (x-pos_central > 0) else 0
    ini_y = y-pos_central if (y-pos_central > 0) else 0
    # indice del elemento máximo dentro de la ventana
    maximo = np.argmax(matriz[ini_x:ini_x+tam_ventana, ini_y:ini_y+tam_ventana])
    # Posición central en la ventrana
    pos_max = math.floor((tam_ventana*tam_ventana)/2)

    return   pos_max == maximo




# Función que hace que cambie los valores de una matriz a 0 dado un tamaño.
def cambiarACero(mat_binaria,tam_ventana,pos_x,pos_y):
    # Calculamos inicio y final en ambas direcciones.
    ini_x = pos_x-tam_ventana if(pos_x-tam_ventana>=0) else 0
    ini_y = pos_y-tam_ventana if(pos_y-tam_ventana>=0) else 0
    final_x = pos_x+tam_ventana+1 if(pos_x+tam_ventana<mat_binaria.shape[0]) else mat_binaria.shape[0]
    final_y = pos_y+tam_ventana+1 if(pos_y+tam_ventana<mat_binaria.shape[1]) else mat_binaria.shape[1]

    mat_binaria[ini_x:final_x,ini_y:final_y] = 0

# Función para suprimir no-máximos.
def suprimirNoMaximos(matriz,tam_ventana,escala):
    mat_binaria=np.full(matriz.shape,255)
    puntosHarris = []

    # Recorremos la mátriz binaria y por cada punto que sea 255, comprobamos si es máximo local o no.
    for i in range(mat_binaria.shape[0]):
        for j in range(mat_binaria.shape[1]):
            if(mat_binaria[i,j] == 255):
                if(comprobarMaximoCentral(matriz,tam_ventana,i,j) ):
                    cambiarACero(mat_binaria,tam_ventana,i,j)
                    puntosHarris.append([matriz[i,j],i,j])


    return puntosHarris

# Función que devuelve los 500 puntos harrys de mayor valor.
def ordenarHarrys(puntosHarrys):
    puntos = []

    for i in range(len(puntosHarrys)):
        max=int(len(puntosHarrys[i])/2) if (int(len(puntosHarrys[i])/2)) < 500 else int(500/2*(i+1))
        sortedHarrys = np.array(sorted(puntosHarrys[i], key=lambda value: value[0], reverse=True))
        puntos.append(sortedHarrys[:max, 1:] )

    return puntos

# Función para dibujar los circulos a los puntos Harrys en la imagen.
def dibujarPuntos(puntos,imagen,orientaciones=[]):
    img = copy.deepcopy(imagen)

    # Transformamos las coordenadas de los puntos harrys en el tamaño original.
    puntosOriginal = []
    puntosOriginal.append(np.array(puntos[0],dtype=int))
    for i in range(1,len(puntos)):
        puntosOriginal.append(np.array(puntos[i] * pow(2,i), dtype=int))


    for i in range(len(puntosOriginal)):
        for indices in puntosOriginal[i]:
            cv2.circle(img, (indices[1],indices[0]) , radius=(i+1), color=1, thickness=-1)

    if(len(orientaciones) > 0):
        for i in range(len(puntosOriginal)):
            radius=(i+1)
            for j in range(len(orientaciones)):
                p = puntosOriginal[i][j]
                angulo= orientaciones[i][j]*180/np.pi
                cv2.line(img,pt1=(p[1],p[0]),
                         pt2=(p[1]+math.floor(np.sin(angulo)*radius), p[0]+math.floor(np.cos(angulo)*radius) ),
                         color=255)


    return img

# Función que engloba el proceso de obtención de puntos harrys.
def obtenerPuntosHarrys(listaEscalas,blocksize,aperturesize):
    listaEigen = []
    # Obtenemos las listas de eigenvalues de cada una de las imagenes.
    for i in listaEscalas:
        listaEigen.append(cv2.cornerEigenValsAndVecs(i,blocksize,aperturesize,cv2.BORDER_DEFAULT) )

    listaIntensidad = []
    for i in listaEigen:
        listaIntensidad.append(calculateHarrysMatrix(i))

    listaMaximos = []
    esc = 1
    for i in listaIntensidad:
        listaMaximos.append(np.array(suprimirNoMaximos(i,tam_ventana=5,escala=esc)))
        esc += 1

    bestHarrys =  ordenarHarrys(listaMaximos)

    return bestHarrys

# Función para optimizar la posición de los puntos.
def refinarPuntos(puntosHarrys,piramide):
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_COUNT, 10, 0.01)
    ref_puntos = []

    for i in range(len(puntosHarrys)):
        ref = np.array(puntosHarrys[i],np.float32)
        cv2.cornerSubPix(image=piramide[i],corners=ref, winSize=(5,5), zeroZone=(-1,-1), criteria=criteria)

        ref_puntos.append(np.array(puntosHarrys[i],np.int) )

    return ref_puntos

# Función para calcular la orientación.
def calcularOrientacion(imagenes,harrys,sigma=5):
    orientaciones = []

    for i in range(len(imagenes)):
        intharrys = np.array(harrys[i],np.int)
        sobelx=cv2.Sobel(imagenes[i],-1,1,0,ksize=sigma )
        sobely=cv2.Sobel(imagenes[i],-1,0,1,ksize=sigma )

        indices = [intharrys[:,0],intharrys[:,1]]

        gradY = sobely[indices[0],indices[1]]
        gradX = sobelx[indices[0],indices[1]]

        orientaciones.append(np.arctan2(gradY,gradX))

    return orientaciones

def computarDescriptoresSIFT():
    detector_sift = cv2.xfeatures2d.SIFT_create()

#-----------------------------------------------------------------------------------------------------------------------
# Prueba lectura de imagen.
yosemite = readImage("imagenes/yosemite1.jpg")
imags = []
createGuassianPyramid(yosemite,imagenes=imags,niveles=5);

blockSize=2
apertureSize=1


puntos=obtenerPuntosHarrys(listaEscalas=imags,blocksize=blockSize,aperturesize=apertureSize)
imagenConHarrys=dibujarPuntos(puntos=puntos,imagen=imags[0])
plt.imshow(imagenConHarrys,'gray')
plt.title("imagen con Harrys sin refinar")
plt.show()

puntosRefinados=refinarPuntos(puntos,imags)
imagenRefinada=dibujarPuntos(puntosRefinados,imags[0])
plt.imshow(imagenRefinada,'gray')
plt.title("imagen con Harrys refinada")
plt.show()


orientaciones=calcularOrientacion(imags,puntosRefinados)
imagenOrienta=dibujarPuntos(puntosRefinados,imags[0],orientaciones)
plt.imshow(imagenOrienta,'gray')
plt.title("imagen con orientaciones")
plt.show()

detector_sift = cv2.xfeatures2d.SIFT_create()

