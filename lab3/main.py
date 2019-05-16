# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

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
    #print(newMatrix)
    return newMatrix

def convolve(kernel, matrix, x, y):
    temp = 0
    n, m = kernel.shape
    for i in range(n):
        for j in range(m):
            temp += kernel.item((i, j)) * matrix.item((x+i,y+j))
    return temp

image = Image.open("leena512.bmp")
array_image = np.asarray(image,dtype=float)
#print(array_image)
array_image = array_image / 255
n,m = array_image.shape
kernelG = [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]
kernelB = [[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1]]
kernelGauss = np.array(kernelG)
kernelGauss = np.multiply(kernelGauss,1/256)
kernelBordes = np.array(kernelB)
#Test
matrix = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
matrixM = np.array(matrix) #esta wea convierte la wea de arriba en array de numpy
# esta wea muestra que agranda la imagen qla
imagenExtendida = extends(array_image)
#print(imagenExtendida)

newMatrix = np.zeros([n, m], dtype=float)
for i in range(n):
    for j in range(m):
        newMatrix[i][j] = convolve(kernelBordes,imagenExtendida,i,j) * 255
#print(newMatrix)
#img = Image.fromarray(newMatrix)
image = newMatrix.astype(np.uint8)
out1 = Image.fromarray(image)
out1.save('test.png')


'''
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
'''