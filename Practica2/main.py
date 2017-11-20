# Práctica 2 VC.
# Alberto Armijo Ruiz.
# 4º GII.

import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy

"""
Ejercicio 1.
"""
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

# Función que hace que cambie los valores de una matriz a 0 dado un tamaño.
def cambiarACero(mat_binaria,tam_ventana,pos_x,pos_y):
    # Calculamos inicio y final en ambas direcciones.
    ini_x = pos_x-tam_ventana if(pos_x-tam_ventana>=0) else 0
    ini_y = pos_y-tam_ventana if(pos_y-tam_ventana>=0) else 0
    final_x = pos_x+tam_ventana+1 if(pos_x+tam_ventana<mat_binaria.shape[0]) else mat_binaria.shape[0]
    final_y = pos_y+tam_ventana+1 if(pos_y+tam_ventana<mat_binaria.shape[1]) else mat_binaria.shape[1]

    # Ponemos su valor a 0.
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
            cv2.circle(img, (indices[1],indices[0]) , radius=(2*i+1), color=1, thickness=-1)

    if(len(orientaciones) > 0):
        for i in range(len(puntosOriginal)):
            radius=(2*i+1)
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
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
        sobelx = convolucionPrimeraDerivada(copy.deepcopy(imagenes[i]),sigma,1,0)
        sobely = convolucionPrimeraDerivada(copy.deepcopy(imagenes[i]),sigma,0,1)

        indices = [intharrys[:,0],intharrys[:,1]]

        gradY = sobely[indices[0],indices[1]]
        gradX = sobelx[indices[0],indices[1]]

        orientaciones.append(np.arctan2(gradY,gradX))

    return orientaciones

# Función para crear el vector de keypoints.
def createKeypoints(puntos,orientaciones):
    # vector de keypoints
    kp = []

    for i in range(len(puntos)):
        px = puntos[i][:,0]
        py = puntos[i][:,1]
        ang = orientaciones[i][:]
        size= 2*i + 1

        for j in range(len(px)):
            kp.append(cv2.KeyPoint(x=px[j],y=py[j],_angle=ang[j],_size=size))


    return kp

# Función para calcular los descriptores.
def calculateDescriptors(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp,descriptors = sift.detectAndCompute(gray,None)

    sol=[kp,descriptors]
    return sol

"""
Ejercicio 2.
"""
# Función para calcular correspondencias por cross-check.
def correspondenciasCrosschek(kp1,desc1,kp2,desc2):
    # declaramos el matcher, en este caso, debemos poner la opción croosCheck a True.
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2,crossCheck=True)
    # calculamos las correspondencias.
    correspondencias=matcher.match(desc1,desc2)

    # Cogemos solamente una parte de las correspondencias las cuales tienen distancia mínima para evitar
    # correspondencias falsas.
    bestCorr = sorted(correspondencias,key=lambda dist:dist.distance, reverse=True)[:50]

    return bestCorr

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

"""
Ejercicio 3.
"""
# Función para borrar los bordes negros de la imagen.
def crop(img):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgray = cv2.medianBlur(imgray,3)

    _, th2, = cv2.threshold(imgray, 8, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = -1
    best_cnt = None

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    approx = cv2.approxPolyDP(best_cnt, 0.01 * cv2.arcLength(best_cnt, True), True)

    far = approx[np.product(approx, 2).argmax()][0]

    ymax = approx[:, :, 0].max()
    xmax = approx[:, :, 1].max()

    x = far[0]
    y = far[1]

    # Crop with the largest rectangle
    crop = img[:y,:x]

    return crop

# Función para calcular un mosaico con 3 imágenes.
# Tendremos en cuenta que en el mosaico las imágenes irán ordenadas.
def mosaico2Imgs(img1,img2):
    # calcalamos descriptores y keypoints
    kp1,desc1 = calculateDescriptors(img1)
    kp2,desc2 = calculateDescriptors(img2)

    size = (int(img1.shape[1]*2),int(img1.shape[0]*2))
    homography = np.identity(3,dtype=np.float32)

    # Pasamos la imagen 1 al mosaico
    result_0 = cv2.warpPerspective(src=img1,M=homography,dsize=size)


    # caculamos correspondencias entre las imágenes.
    corr1_2 = correspondencias2NN(kp1,desc1,kp2,desc2)
    # calculamos homografías a partir de las correspondencias.
    src_pts1 = np.float32([ kp1[m[0].queryIdx].pt for m in corr1_2 ]).reshape(-1,1,2)
    dst_pts1 = np.float32([ kp2[m[0].trainIdx].pt for m in corr1_2 ]).reshape(-1,1,2)

    homography1, mask1 = cv2.findHomography(srcPoints=src_pts1,dstPoints=dst_pts1,method=cv2.RANSAC,ransacReprojThreshold=1.0)

    # Calculamos homografía inversa.
    homography1 = np.linalg.inv(np.matrix(homography1))
    homography1 = homography * homography1

    result = cv2.warpPerspective(src=img2,M=homography1,dsize=size,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    result[0:img1.shape[0],0:img1.shape[1]]=img1

    result = crop(result)


    return result

# Función para crear mosaico con 3 imágenes.
def mosaico3Img(img1,img2,img3):
    # Separamos las imágenes en dos grupos y creamos sus mosaicos
    mosaico_dch = mosaico2Imgs(img1,img2)
    # Unimos las imágenes en un único mosaico.
    mosaico = mosaico2Imgs(mosaico_dch,img3)

    return mosaico

"""
Ejercicio 4.
"""
# Función para crear mosaico con N imágenes.
def mosaicoNImg(imgs):
    # hay dos ó tres imágenes, llamamos a las funciones ya construidas.
    if len(imgs) == 2:
        mosaico=mosaico2Imgs(imgs[0],imgs[1])
    elif len(imgs) == 3:
        mosaico = mosaico3Img(imgs[0],imgs[1],imgs[2])
    else:
        # Calculamos la mitad, y creamos un mosaico para cada mitad.
        # Si la mitad es demasiado grande se vuelve a llamar a las funciones.
        mitad = int(len(imgs)/2)
        dch=mosaicoNImg(imgs[:mitad])
        izq=mosaicoNImg(imgs[mitad:])
        mosaico = mosaico2Imgs(dch,izq)

    return mosaico




#-----------------------------------------------------------------------------------------------------------------------
# Prueba lectura de imagen.
yosemite = readImage("imágenes/yosemite1.jpg")
imags = []
createGuassianPyramid(yosemite,imagenes=imags,niveles=5)

blockSize=2
apertureSize=1

"""
Ejercicio 1.
"""
### Apartado A.
puntos=obtenerPuntosHarrys(listaEscalas=imags,blocksize=blockSize,aperturesize=apertureSize)
imagenConHarrys=dibujarPuntos(puntos=puntos,imagen=imags[0])
plt.imshow(imagenConHarrys,'gray')
plt.title("imagen con puntos Harry's")
plt.show()

### Apartado B.
puntosRefinados=refinarPuntos(puntos,imags)
imagenRefinada=dibujarPuntos(puntosRefinados,imags[0])
plt.imshow(imagenRefinada,'gray')
plt.title("imagen con puntos refinados.")
plt.show()


### Aparatdo C.
orientaciones=calcularOrientacion(imags,puntosRefinados)
imagenOrienta=dibujarPuntos(puntosRefinados,imags[0],orientaciones)
plt.imshow(imagenOrienta,'gray')
plt.title("imagen con orientaciones")
plt.show()

### Apartado D.
img=cv2.imread("imágenes/yosemite1.jpg")
kp = createKeypoints(puntos,orientaciones)
keyPoints,descriptors = calculateDescriptors(img)
img2 = copy.deepcopy(img)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.drawKeypoints(img,keyPoints,outImage=img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0,255,0))
plt.imshow(img2,'gray')
plt.title("imagen con keypoints dibujados.")
plt.show()

"""
Ejercicio 2.
"""
yosemite2 = cv2.imread("imágenes/yosemite2.jpg")
kp2,desc2 = calculateDescriptors(yosemite2)
yosemite2 = cv2.cvtColor(yosemite2,cv2.COLOR_BGR2GRAY)

crosscheck_corr = correspondenciasCrosschek(keyPoints,descriptors,kp2,desc2)
crosscheck_image = cv2.drawMatches(img1=img,keypoints1=keyPoints,img2=yosemite2,keypoints2=kp2,matches1to2=crosscheck_corr,outImg=None)
plt.imshow(crosscheck_image,'gray')
plt.title("Correspondencias Cross-Check")
plt.show()

knn_corr = correspondencias2NN(keyPoints,descriptors,kp2,desc2)
knn_image = cv2.drawMatchesKnn(img1=img,keypoints1=keyPoints,img2=yosemite2,keypoints2=kp2,matches1to2=knn_corr,outImg=None)
plt.imshow(knn_image,'gray')
plt.title("Correspondencias 2NN")
plt.show()

"""
Ejercicio 3.
"""
yosemite1 = cv2.imread("imágenes/yosemite1.jpg")
yosemite2 = cv2.imread("imágenes/yosemite2.jpg")
yosemite3 = cv2.imread("imágenes/yosemite3.jpg")


mosaico = mosaico3Img(yosemite1,yosemite2,yosemite3)
mosaico = cv2.cvtColor(mosaico,cv2.COLOR_BGR2RGB)
plt.imshow(mosaico)
plt.title("mosaico de 3 imágenes")
plt.show()

"""
Ejercicio 4.
"""
mos1 = cv2.imread("imágenes/mosaico002.jpg")
mos2 = cv2.imread("imágenes/mosaico003.jpg")
mos3 = cv2.imread("imágenes/mosaico004.jpg")
mos4 = cv2.imread("imágenes/mosaico005.jpg")
mos5 = cv2.imread("imágenes/mosaico006.jpg")
mos6 = cv2.imread("imágenes/mosaico007.jpg")
mos7 = cv2.imread("imágenes/mosaico008.jpg")
mos8 = cv2.imread("imágenes/mosaico009.jpg")
mos9 = cv2.imread("imágenes/mosaico010.jpg")
mos10 = cv2.imread("imágenes/mosaico011.jpg")

imgs=[mos1,mos2,mos3,mos4,mos5,mos6,mos7,mos8,mos9,mos10]

mosaico = mosaicoNImg(imgs)
mosaico = cv2.cvtColor(mosaico,cv2.COLOR_BGR2RGB)
plt.imshow(mosaico)
plt.title("mosaico de N imágenes")
plt.show()
