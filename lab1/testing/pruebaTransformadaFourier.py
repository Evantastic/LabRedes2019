# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.fftpack import fft, fftfreq, ifft
from scipy.io.wavfile import read, write

# File
file = read("../resources/handel.wav")
rate = file[0]
amplitudeTime = file[1]
dataLen = len(amplitudeTime)
duration = dataLen/rate
delta = 1/rate
time = np.linspace(0, (dataLen - 1) * delta, dataLen)
amplitudeFrequency = fft(amplitudeTime)
frequency = fftfreq(dataLen, delta)
inverseAmplitude = ifft(amplitudeFrequency).real
fakedAmplitude = amplitudeFrequency.copy()
for x in range(12504,36557):
    fakedAmplitude[x] = 0;
    fakedAmplitude[73113 - x] = 0
for x in range(2000):
    fakedAmplitude[x] = 0;
    fakedAmplitude[73112 - x] = 0
inverseFakedAmplitude = ifft(fakedAmplitude).real
firstQuadraticError = np.sqrt(((inverseAmplitude - amplitudeTime) ** 2).mean())
secondQuadraticError = np.sqrt(((inverseFakedAmplitude - amplitudeTime) ** 2).mean())
write("../resources/handelTruncado.wav",rate,inverseFakedAmplitude)

# # Gráfico de la transformada de fourier
# plt.subplot(2,3,1)
# plt.subplot(2,1,1)
# plt.vlines(frequency, 0, np.abs(amplitudeFrequency))
# plt.title("Transformada de Fourier")
# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Amplitud')
# plt.subplot(2,1,2)
# plt.vlines(frequency, 0, np.abs(fakedAmplitude))
# plt.title("Transformada de Fourier truncada")
# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Amplitud')
# # Gráfico de la transformada de fourier truncada
# plt.subplot(2,3,2)
# plt.vlines(frequency, 0, np.abs(amplitudeFrequency))
# plt.title("Transformada de Fourier truncada")
# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Amplitud')
# Grafico de la señal
# plt.subplot(2,1,1)
# plt.plot(time, amplitudeTime,"r")
# plt.title("Señal original")
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud')
# plt.subplot(2,1,2)
# plt.plot(time, inverseFakedAmplitude,"r")
# plt.title("Señal inversa")
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud') 
# Grafico de la señal obtenida a traves de la transformada inversa
# plt.subplot(2,3,4)
# plt.plot(time, inverseAmplitude,"b")
# plt.title("Señal obtenida a través de la transformada de Fourier inversa")
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud')
# # Grafico de la señal obtenida a traves de la transformada inversa de la señal truncada
# plt.subplot(2,3,5)
# plt.plot(time, inverseAmplitude,"b")
# plt.title("Señal obtenida a través de la transformada de Fourier inversa con frecuencias truncadas")
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud')

# plt.show()
