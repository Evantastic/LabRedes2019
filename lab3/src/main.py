# -*- coding: utf-8 -*-
import numpy as np

def extends(matrix):
    n, m = matrix.shape
    newMatrix = np.zeros([n + 4, m + 4], dtype=int)
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
    print(newMatrix)

def convolve(kernel, matrix, x, y):
    temp = 0
    n, m = kernel.shape
    for i in range(n):
        for j in range(m):
            temp += kernel.item((i, j)) + matrix.item((x+i,y+j))
    return temp

kernel = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]] 
matrix = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]] 
kernel = np.asmatrix(kernel)
matrix = np.asmatrix(matrix)
print(convolve(kernel, matrix, 0, 0))
extends(matrix)
