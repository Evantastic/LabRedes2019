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
plt.figure(1)
plt.plot(time,amplitudeTime)
plt.title("Señal original")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.figure(2)
plt.plot(time,inverseAmplitude)
plt.title("Señal invertida")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.figure(3)
plt.plot(time,inverseFakedAmplitude)
plt.title("Señal invertida truncada")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.figure(4)
plt.vlines(frequency, 0, np.abs(amplitudeFrequency))
plt.title("Transformada de Fourier")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.figure(5)
plt.vlines(frequency, 0, np.abs(fakedAmplitude))
plt.title("Transformada de Fourier truncada")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')

plt.show()
