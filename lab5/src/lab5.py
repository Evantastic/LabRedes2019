import matplotlib.pyplot as plt
import numpy as np
from random import choice

#
# Modulacion de una senal a partir de bits
# Recibe un bit de una senal, la frecuencia, tasa de datos y numero de puntos
# Multiplica un bit de la senal por la portadora
# Retorna el bit modulado
def modulator(bit, frequency, rate, points):
    t = 0
    ts = np.linspace(t, t+rate,points)
    t+=rate
    return 50*bit*np.cos(frequency*ts)

#
# Demodulacion de una senal a partir de los valores del arreglo de modulacion
# Recibe el arreglo de senal modulada y el numero de puntos de esta
# Recorre el arreglo y verifica los valores de cada punto de la senal
# Si los valores corresponden a 0, originalmente venia un 0
# Si los valores corresponden a otro valor, originalmente venia un 1
# Retorna la senal demodulada, es decir, la original
def demodulator(array,points):
    aux = 0
    outputBits = ''
    while (aux < len(array)):
        if(array[aux] == 0.0 or array[aux] == -0.0):
            outputBits += '0'
        else:
            outputBits += '1'
        aux = aux + points + 1
    return outputBits

#
# Agregacion de ruido a una senal dado un determinado SNR
# Recibe la senal a la que se le agregara ruido y en snr determinado para realizarlo
# Retorna la senal con ruido agregado
def noise(signal,snr):
    noise = np.random.normal(0.0,1.0/snr,len(signal))
    noiseSignal = noise + signal
    return noiseSignal

#
# Identificacion de tasa de error entre senal original y modulada
# Determina la tasa de error entre dos senales demoduladas, una demodulada normalmente y la otra con cierto nivel de ruido
# Se recorre la senal demodulada y la senal demodulada con ruido para realizar una comparacion bit a bit, si son diferentes se incrementa el
# valor de errores encontrados
# Se retorna la tasa de error como los errores encontrados por el largo de bits recorridos
def errorT(demodulated,demodulatedN):
    error = 0
    for i in range(len(demodulated)):
        if demodulated[i] != demodulatedN[i]:
            error += 1
    errorRate = error/len(demodulated)
    return errorRate

#
# Se solicita el numero de bits a generar la senal aleatoria
bitQuantity = int(input("Ingrese el numero de bits: "))
# Se considera una frecuencia de muestreo igual a 100
frequency = 100
#Se considera un numero de puntos considerando el teorema de muestreo y un rate de 0.1
points = frequency//2
rate = 0.1
#Se obtiene el largo de los bits como el numero de puntos por bit multiplicado con la cantidad de bits
bitsLen = int(points*bitQuantity)
#Se genera la senal pseudoaleatoria de bits de largo n ingresado como parametro
bitString = ''.join(choice(['0', '1']) for _ in range(bitQuantity))
#Se transforma el string de bits a un arreglo
bitArray = list(bitString)

#
#Se modula la senal recorriendo cada bit pseudoaleatorio y se modula por bit, se anade el resultado a un arreglo
array = []
for c in bitArray:
    array.extend(modulator(int(c), frequency*2*np.pi, rate, points))

#
# A continuación viene el ploteo de la senal modulada, junto con los bits de la senal original
time = np.linspace(0, bitQuantity*rate, points*bitQuantity)
plt.plot(time, array)
plt.title("Senal original : "+bitString)
plt.ylabel("Amplitud")
plt.xlabel("Tiempo (s)")
plt.show()

# A continuacion vienen los bits obtenidos desde la senal, los cuales son demodulados e identifica cuales eran originalmente
bitsDemodulated = demodulator(array,points)
print("Demodulacion: La senal es "+bitsDemodulated)

#Se le agrega ruido a la senal con un valor de prueba SNR = 4
snrValue = int(input("Ingrese un valor de prueba de SNR: "))
noisySignal = noise(array,snrValue)

# A continuación viene el ploteo de la senal con ruido
plt.plot(time,noisySignal)
plt.title("Senal con ruido SRN = "+str(snrValue))
plt.ylabel("Amplitud")
plt.xlabel("Tiempo (s)")
plt.show()

# Se considera un arreglo con 6 niveles de SNR y un arreglo para almacenar los errores
SNR = [1,2,4,6,8,10]
errors = []
# Para cada SNR se le agrega un ruido de SNR a la senal modulada, se demodula la senal con ruido, se verifica el error y se agregan al arreglo errores
for snr in SNR:
    sModulated = noise(array,snr)
    sDemodulated = demodulator(sModulated,points)
    error = errorT(bitsDemodulated,sDemodulated)
    errors.append(error)
#Se muestra por consola los valores de BER y SNR
print("SNR: "+str(SNR))
print("Errores: "+str(errors))
#Se muestra el grafico BER vs SNR
plt.plot(errors,SNR)
plt.title("BER vs SNR")
plt.show()
