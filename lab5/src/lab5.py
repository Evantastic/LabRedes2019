import matplotlib.pyplot as plt
import numpy as np
import random
from random import choice

#
# Modulacion de una senal a partir de bits
#
def modulator(bit, frequency, rate, points):
    t = 0
    ts = np.linspace(t, t+rate,points)
    t+=rate
    return 50*bit*np.cos(frequency*ts)

#
# Demodulacion de una senal a partir de los valores del arreglo de modulacion
#
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
#
def noise(signal,snr):
    noise = np.random.normal(0.0,1.0/snr,len(signal))
    noiseSignal = noise + signal
    return noiseSignal

#
# Identificacion de tasa de error entre senal original y modulada
#
def error(demodulated,demodulatedN):
    error = 0
    for i in range(len(demodulated)):
        if demodulated[i] != demodulatedN[i]:
            error += 1
    errorRate = error/len(demodulated)
    return errorRate

#
#Generacion de senal y bits
#
bitQuantity = 5
frequency = 100
#Teorema de muestreo
points = frequency//2
rate = 0.1
bitsLen = int(points*bitQuantity)
bitString = ''.join(choice(['0', '1']) for _ in range(bitQuantity))
bitArray = list(bitString)

#
#Se modula
#
array = []
for c in bitArray:
    array.extend(modulator(int(c), frequency*2*np.pi, rate, points))


# A continuación viene el ploteo de la senal modulada
time = np.linspace(0, bitQuantity*rate, points*bitQuantity)
plt.plot(time, array)
plt.title("Senal original : "+bitString)
plt.show()

# A continuacion vienen los bits obtenidos desde la senal
bitsDemodulated = demodulator(array,points)
print("La senal es "+bitsDemodulated)

#Senal con ruido
snrValue = 4
noisySignal = noise(array,snrValue)

# A continuación viene el ploteo de la senal con ruido
plt.plot(time,noisySignal)
plt.title("Senal con ruido SRN = "+str(snrValue))
plt.show()

# Se generan n random bits
randomBits = []
for i in range(10):
    bit = str(random.randint(0,1))
    randomBits.append(bit)
    ''.join(bitString)

'''Falta:
Simule la transmisión de los bits aplicando el modulador, demodulador y el canal AWGN para varios niveles ruido.
Considere al menos 6 niveles de SNR entre -2 y 10 dB.
Para cada SNR determine la tasa de errores de bit (BER) de su demodulador comparando los bits demodulados con los bits originales.'''
