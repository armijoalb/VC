# Trabajo 1 Visión por Computador.
# Alberto Armijo Ruiz.
# 4 GII.

import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
import copy

plt.interactive(False)

def readImage(pathToImage):
    imagen = cv2.imread(pathToImage)
    imagen = cv2.cvtColor(src=imagen, code=cv2.COLOR_BGR2GRAY)
    return imagen

# Ejercicio 1.
# Apartado A.
"""
Con imágenes, 
"""
def mostrarMultiplesImagenes(imagenes, titulos):
    # Calculamos el número de filas y columnas en las que tenemos que dividir
    # nuestro recuadro. Si es entero, en n*n cuadrados, sino en (n+1)^2 cuadrados.s
    n = math.sqrt(len(imagenes))
    fil =  col = n
    
    if( n - int(n) > 0 ):
        fil = col = n+1

    # pintamos las imágenes.
    for i in range(len(imagenes)):
        img = imagenes[i]
        plt.subplot(fil, col, i+1)
        plt.title(titulos[i])
        plt.imshow(img,'gray')
        
    plt.show()
    

# Apartado B.
def convolucionGaussiana(imagen, sigma,tipoBorde=0):
    # Tamaño del kernel.
    tam = 6*sigma +1
    # Kernel separado.
    sepGaussainKernel = cv2.getGaussianKernel(tam, sigma)
    # Kernel completo.
    gaussianKernel = sepGaussainKernel * np.transpose(sepGaussainKernel)

    # Realizamos la convolución.
    if tipoBorde == 0:
        convImag = cv2.filter2D(imagen,-1, gaussianKernel,cv2.BORDER_CONSTANT)
    else:
        convImag = cv2.filter2D(imagen,-1,gaussianKernel,cv2.BORDER_REPLICATE)

    # Devolvemos el resultado.
    return convImag

# Apartado C.
def convolucionSeparada(imagen, sigma,tipoBorde=0):
    # Calculamos tamaño máscara y obtenemos el kernel.
    tam = 6*sigma +1
    sepKernel = cv2.getGaussianKernel(tam, sigma)

    # Filas y columnas de la imagen.
    height, width = imagen.shape

    convImg = imagen

    # Recorremos filas y columnas.
    if tipoBorde == 0:
        convImg[0:height]=cv2.filter2D(imagen[:],-1,sepKernel,cv2.BORDER_CONSTANT)
        convImg[:,0:width]=cv2.filter2D(imagen[:,0:width],-1,sepKernel.T,cv2.BORDER_CONSTANT)
    else:
        convImg[0:height] = cv2.filter2D(imagen[:], -1, sepKernel, cv2.BORDER_REPLICATE)
        convImg[:, 0:width] = cv2.filter2D(imagen[:, 0:width], -1, sepKernel.T, cv2.BORDER_REPLICATE)

    return convImg

def convolucionPrimeraDerivada(imagen,tam,derivX, derivY,tipoBorde=0):
    # Calculamos derivadas en x e y.
    kx, ky = cv2.getDerivKernels(derivX,derivY,tam)
    convImg = imagen
    # Filas y columnas de la imagen.
    height, width = imagen.shape

    # Aplicamos filtros a la imagen.
    if tipoBorde == 0:
        convImg[:] = cv2.filter2D(imagen[:], -1, kx.T, cv2.BORDER_CONSTANT)
        convImg[:, :width] = cv2.filter2D(imagen[:, :width], -1, ky, cv2.BORDER_CONSTANT)
    else:
        convImg[:] = cv2.filter2D(imagen[:], -1, kx.T, cv2.BORDER_REPLICATE)
        convImg[:, :width] = cv2.filter2D(imagen[:, :width], -1, ky, cv2.BORDER_REPLICATE)

    # Devolvemos el resultado.
    return convImg

def convolucionSegundaDerivada(imagen, tam, derivX, derivY,tipoBorde=0):
    convImg=convolucionPrimeraDerivada(imagen, tam, 2*derivX, 2*derivY,tipoBorde)
    return convImg

def convolucionNucleoLaplaciano(imagen, tam,tipoBorde=0):
    # Obtenemos la segunda derivada en X.
    convImgX = convolucionSegundaDerivada(copy.deepcopy(imagen), tam, 1,0,tipoBorde)
    # Obtenemos la segunda derivada en Y.
    convImgY = convolucionSegundaDerivada(copy.deepcopy(imagen), tam, 0,1,tipoBorde)

    # Obtenemos la imagen de la laplaciana.
    convImg = convImgX + convImgY
    # Devolvemos el resultado.
    return convImg

def createGuassianPyramid(imagen,tipoBorde=0):
    # Primero crearemos la imagen que vamos a devolver.
    height,width=imagen.shape
    total_height = int(height)
    total_width = int(1.5*width+1)

    # Imagen de la pirámide.
    gaussianPyramid=np.ones((total_height,total_width))

    # Pegamos la primera imagen dentro de la imagen original.
    gaussianPyramid[0:height,0:width]=imagen


    # Reducimos el tamaño de la imagen.
    new_image = cv2.pyrDown(imagen)
    aux_h = 0

    # Aplicamos el mismo proceso por cada uno de los niveles siguientes de la pirámide.
    for i in range(0,4):

        gaussianPyramid[aux_h:aux_h+new_image.shape[0],width+1:width+1+new_image.shape[1]]=new_image
        aux_h = aux_h+new_image.shape[0] + 1

        new_image = cv2.pyrDown(new_image)


    return gaussianPyramid

def createLaplacianDiference(imagen, tipoBorde=0):
    # Función que devuelve la laplaciana de la imagen para realizar la pirámide laplaciana.
    original = copy.deepcopy(imagen)
    size = (original.shape[1], original.shape[0])

    new_img=cv2.pyrDown(original,borderType=cv2.BORDER_DEFAULT)
    new_img=cv2.pyrUp(new_img, dstsize =  size, borderType=cv2.BORDER_DEFAULT)
    lap=cv2.subtract(original,new_img)

    return lap

def createLaplacianPyramid(imagen,tipoBorde=0):
    # Calculamos dimensiones de la pirámide.
    total_height=int(imagen.shape[0])
    total_width=int(1.5*imagen.shape[1]+1)

    # Creamos imagen contenedora.
    laplacianPyramid = np.ones((total_height,total_width),np.float)

    # Creamos el primer nivel y lo pegamos en la imagen de salida.
    orig=copy.deepcopy(imagen)
    lap=createLaplacianDiference(copy.deepcopy(orig),tipoBorde)
    height,width=lap.shape
    laplacianPyramid[0:height, 0:width] = lap

    orig = cv2.pyrDown(orig, borderType=cv2.BORDER_DEFAULT)

    aux_h = 0

    # Repetimos el proceso para cada una de los siguientes niveles.
    for i in range(0,4):

        lap=createLaplacianDiference(copy.deepcopy(orig),tipoBorde)
        laplacianPyramid[aux_h:aux_h+lap.shape[0],width+1:width+1+lap.shape[1]]=lap

        aux_h = aux_h+lap.shape[0] + 1

        orig = cv2.pyrDown(orig, borderType=cv2.BORDER_DEFAULT)



    return laplacianPyramid

def hibridarImagenes(imagen1, imagen2,sigma,alpha):
    # Vector de imagenes.
    imagenes = []
    # Obtenemos frecuencias altas.
    img_frec_altos=convolucionSeparada(imagen1,sigma)
    # Obtenemos frecuencias bajas.
    img_frec_bajas=convolucionSeparada(imagen2,sigma+3)

    # Introducimos las imágenes dentro del vector.
    imagenes.append(img_frec_altos)
    imagenes.append(img_frec_bajas)

    beta=1-alpha

    # Calculamos la función híbrida y la introducimos.
    hybrid=cv2.addWeighted(img_frec_altos,alpha,img_frec_bajas,beta,0)
    imagenes.append(hybrid)

    return imagenes

def mostrarResultadoHibridacion(img1,img2,dst):
    # Creamos un grid.
    gs = gridspec.GridSpec(2, 2)

    # Colocamos la imagen con frecuencias altas.
    plt.subplot(gs[0, 0])
    plt.imshow(img1, 'gray')
    # Colocamos la imagen con frecuencias bajas.
    plt.subplot(gs[1, 0])
    plt.imshow(img2, 'gray')
    # Colocamos la imagen híbrida.
    plt.subplot(gs[:, 1])
    plt.imshow(dst, 'gray')
    # Mostramos el resultado.
    plt.show()





image = cv2.imread('imagenes/messi.jpg',0)
image = image.astype(float)
image = cv2.normalize(image,image,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

sig = 8
tam = 6*sig+1

sig2 = 2
tam2= 6*sig2 +1

#imagen difuminada. Apartado B
difuminada=cv2.GaussianBlur(image,(tam,tam),sig,borderType=cv2.BORDER_CONSTANT)
difuminada2=cv2.GaussianBlur(image,(tam2,tam2),sig2,borderType=cv2.BORDER_REPLICATE)
mi_difuminada=convolucionGaussiana(copy.deepcopy(image),sig)
mi_difuminada2=convolucionSeparada(copy.deepcopy(image),sig2,1)
mostrarMultiplesImagenes(imagenes=[difuminada,difuminada2,mi_difuminada,mi_difuminada2],
                         titulos=['imagen con GaussianBlur 1','imagen con GaussianBlur 2', 'ejemplo 1', 'ejemplo 2'])

#imagen difumindad convolución separada. Apartado C
separada=convolucionSeparada(copy.deepcopy(image),sig)
separada2=convolucionSeparada(copy.deepcopy(image),sig2,1)
mostrarMultiplesImagenes(imagenes=[difuminada,separada,difuminada2,separada2],
                         titulos=['imagen con GaussianBlur 1', 'ejemplo 1', 'imagen con GuassianBlur 2', 'ejemplo 2'])

#imagen de la primera derviada. Apartado D.
# Eje X.
new_sig=1
new_tam=6*new_sig+1
sobelx=cv2.Sobel(image,-1,1,0,ksize=new_tam, borderType=cv2.BORDER_CONSTANT)
sobelx2=cv2.Sobel(image,-1,1,0,ksize=1, borderType=cv2.BORDER_REPLICATE)
derivX=convolucionPrimeraDerivada(copy.deepcopy(image),new_tam,1,0)
derivX2=convolucionPrimeraDerivada(copy.deepcopy(image),1,1,0,1)
mostrarMultiplesImagenes(imagenes=[sobelx,derivX,sobelx2,derivX2],
                         titulos=['imagen con Sobel 1 en X','ejemplo 1','imagen con Sobel 2 en X','ejemplo 2'])

#Eje Y.
sobely=cv2.Sobel(image,-1,0,1,ksize=new_tam,borderType=cv2.BORDER_CONSTANT)
derivY=convolucionPrimeraDerivada(copy.deepcopy(image),new_tam,0,1)
sobely2=cv2.Sobel(image,-1,0,1,ksize=1,borderType=cv2.BORDER_REPLICATE)
derivY2=convolucionPrimeraDerivada(copy.deepcopy(image),1,0,1,1)
mostrarMultiplesImagenes(imagenes=[sobely,derivY,sobely2,derivY2],
                         titulos=['imagen con Sobel 1 en Y', 'ejemplo 1', 'imagen con Sobel 2 en Y', 'ejemplo 2'])

#imagen de la segunda derivada. Apartado E
# Eje X.
sobel2x=cv2.Sobel(image,-1,2,0,ksize=new_tam, borderType=cv2.BORDER_CONSTANT)
deriv2X=convolucionSegundaDerivada(copy.deepcopy(image),new_tam,1,0)
sobel2x2=cv2.Sobel(image,-1,2,0,ksize=1,borderType=cv2.BORDER_REPLICATE)
deriv2X2=convolucionSegundaDerivada(copy.deepcopy(image),1,1,0,1)
mostrarMultiplesImagenes(imagenes=[sobel2x,deriv2X,sobel2x2,deriv2X2],
                         titulos=['imagen con segunda derivada en X 1', 'ejemplo 1', 'imagen con segunda derivada en X 2', 'ejemplo 2'])

# Eje Y.
sobel2y=cv2.Sobel(image,-1,0,2,ksize=new_tam,borderType=cv2.BORDER_CONSTANT)
deriv2Y=convolucionSegundaDerivada(copy.deepcopy(image),new_tam,0,1)
sobel2y2=cv2.Sobel(image,-1,0,2,ksize=1,borderType=cv2.BORDER_REPLICATE)
deriv2Y2=convolucionSegundaDerivada(copy.deepcopy(image),1,0,1,1)
mostrarMultiplesImagenes(imagenes=[sobel2y,deriv2Y,sobel2y2,deriv2Y2],
                         titulos=['imagen segunda derivada en Y 1', 'ejemplo 1','imagen segunda derivada en Y 2', 'ejemplo 2'])

#imagen con la laplaciana. Apartado F
laplaciana=cv2.Laplacian(image,-1,ksize=new_tam,borderType=cv2.BORDER_CONSTANT)
mi_lap=convolucionNucleoLaplaciano(copy.deepcopy(image),new_tam)
laplaciana2=cv2.Laplacian(image,-1,ksize=1,borderType=cv2.BORDER_REPLICATE)
mi_lap2=convolucionNucleoLaplaciano(copy.deepcopy(image),1,1)
mostrarMultiplesImagenes(imagenes=[laplaciana,mi_lap,laplaciana2,mi_lap2],
                         titulos=['imagen laplaciana 1','ejemplo 1','imagen laplaciana 2','ejemplo 2'])


# Pirámide Guassiana. Apartado G
piramideGaussianna=createGuassianPyramid(copy.deepcopy(image))
plt.imshow(piramideGaussianna,'gray')
plt.title("pirámide guassiana")
plt.show()

piramideGaussianna2=createGuassianPyramid(copy.deepcopy(image),1)
plt.imshow(piramideGaussianna2,'gray')
plt.title("pirámide guassiana 2")
plt.show()


# Pirámide Laplaciana. Apartado H
lap1=createLaplacianPyramid(copy.deepcopy(image))
plt.imshow(lap1,'gray')
plt.title("Pirámide laplaciana")
plt.show()

lap2=createLaplacianPyramid(copy.deepcopy(image),1)
plt.imshow(lap2,'gray')
plt.title("Piramide laplaciana 2")
plt.show()


# Hibridación de dos imagenes.
# Ejemplo 1.
sig1=2
img1=cv2.imread("imagenes/cat.bmp",0)
cv2.normalize(img1,img1,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
img2=cv2.imread("imagenes/dog.bmp",0)
cv2.normalize(img2,img2,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

alpha=0.3
beta=(1-alpha)
src=hibridarImagenes(copy.deepcopy(img1),copy.deepcopy(img2),sig1,alpha)

mostrarResultadoHibridacion(src[0],src[1],src[2])

# Ejemplo 2.
img1=cv2.imread("imagenes/marilyn.bmp",0)
cv2.normalize(img1,img1,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
img2=cv2.imread("imagenes/einstein.bmp",0)
cv2.normalize(img2,img2,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

alpha=0.25
beta=(1-alpha)
src=hibridarImagenes(copy.deepcopy(img1),copy.deepcopy(img2),sig1,alpha)
mostrarResultadoHibridacion(src[0],src[1],src[2])

# Ejemplo 3.
img1=cv2.imread("imagenes/submarine.bmp",0)
cv2.normalize(img1,img1,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
img2=cv2.imread("imagenes/fish.bmp",0)
cv2.normalize(img2,img2,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)

alpha=0.25
beta=(1-alpha)
src=hibridarImagenes(copy.deepcopy(img1),copy.deepcopy(img2),sig1,alpha)
mostrarResultadoHibridacion(src[0],src[1],src[2])
