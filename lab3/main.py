# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

#Funcion que extiende la matriz original, replicando los pixeles adyacentes y utilizandolos como borde
#Entrada: Imagen representada como un array
#Salida: Imagen extendida con 2 bordes como un array
def extends(matrix):
    n, m = matrix.shape
    newMatrix = np.zeros([n + 4, m + 4], dtype=float)
    for i in range(n):
        for j in range(m):
            newMatrix[i+2,j+2] = matrix.item((i,j))
    for i in range(1, m-1):
        newMatrix[0,i+2] = matrix.item((0,i))
        newMatrix[1,i+2] = matrix.item((0,i))
        newMatrix[n+2,i+2] = matrix.item((n-1,i))
        newMatrix[n+3,i+2] = matrix.item((n-1,i))
    for i in range(1, n-1):
        newMatrix[i+2,0] = matrix.item((i,0))
        newMatrix[i+2,1] = matrix.item((i,0))
        newMatrix[i+2,n+2] = matrix.item((i,n-1))
        newMatrix[i+2,n+3] = matrix.item((i,n-1))
    for i in range(3):
        for j in range(3):
            newMatrix[i,j] = matrix.item((0,0))
            newMatrix[n+1+i,m+1+j] = matrix.item((n-1,m-1))
            newMatrix[i,m+1+j] = matrix.item((0,m-1))
            newMatrix[n+1+i,j] = matrix.item((n-1,0))
    return newMatrix

#Funcion que realiza la convolucion entre dos matrices, retorna el valor correspondiente
#Entrada: Kernel y matriz como array y posicion a calcular la aplicacion del kernel sobre la matriz
#Salida: Valor que posee la convolucion en el punto x,y dado
def convolve(kernel, matrix, x, y):
    temp = 0
    n, m = kernel.shape
    for i in range(n):
        for j in range(m):
            temp += kernel.item((i, j)) * matrix.item((x+i,y+j))
    return temp

#Funcion que obtiene y muestra la transformada de fourier de una imagen segun su tipo
#Entrada: Imagen a la que se le aplicara la transformada como array y el tipo (si es original, gaussiana o con bordes)
#Salida: Graficos de las transformadas de fourier
def fourier2D(image,type):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    plt.figure(type), plt.imshow(magnitude_spectrum, cmap='gray')
    if type == 1:
        plt.title('Transformada de Fourier Imagen Original')
    elif type == 2:
        plt.title('Transformada de Fourier Imagen Gauss')
    else:
        plt.title('Transformada de Fourier Imagen Bordes')
    plt.colorbar()
    plt.show()

#Apertura de la imagen y almacenamiento como array
image = Image.open("leena512.bmp")
array_image = np.asarray(image,dtype=float)
#Normalizacion de la imagen
array_image = array_image / 255
#Obtencion de dimensiones
n,m = array_image.shape
#Definicion de kernel gaussiano y de bordes
kernelG = [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]
kernelB = [[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1]]
#Almacenamiento como array
kernelGauss = np.array(kernelG)
kernelGauss = np.multiply(kernelGauss,1/256)
kernelBordes = np.array(kernelB)
#Se extienden los bordes de la imagen original
imagenExtendida = extends(array_image)
#Se crean los nuevos array para imagenes con filtro
filtroGaussiano = np.zeros([n, m], dtype=float)
filtroBordes = np.zeros([n, m], dtype=float)
print('Procesando imagenes')
#Se recorre la imagen y se aplican los kernels, los valores se almacenan en su array correspondiente
for i in range(n):
    for j in range(m):
        filtroGaussiano[i][j] = convolve(kernelGauss,imagenExtendida,i,j) * 255
        filtroBordes[i][j] = convolve(kernelBordes,imagenExtendida,i,j) * 255
#Transformacion de arreglo a imagen gaussiana
imgGauss = filtroGaussiano.astype(np.uint8)
gauss = Image.fromarray(imgGauss)
gauss.save('Kernel-Gauss.png')
#Transformacion de arreglo a imagen con bordes
imgBordes = filtroBordes.astype(np.uint8)
bordes = Image.fromarray(imgBordes)
bordes.save('Kernel-Bordes.png')
print('Procesamiento finalizado')

#Transformada de Fourier de la imagen original, imagen con kernel gauss e imagen con kernel de bordes
fourier2D(array_image,1)
fourier2D(imgGauss,2)
fourier2D(imgBordes,3)
