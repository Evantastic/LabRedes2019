# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.fftpack import fft, fftfreq
from scipy.io.wavfile import read

# Archivo
archivo = read("../resources/handel.wav")
rate = archivo[0]
datos = archivo[1]
muestras = len(datos)
duracion = muestras/rate
delta = 1/rate

# Datos de la Señal
variableTiempo = np.linspace(0, (muestras - 1) * delta, muestras)

# Grafico de la señal
plt.subplot(2,1,1)
plt.plot(variableTiempo, datos)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (Db)')

# Datos de la Transformada de Fourier
variableY = fft(datos)
variableX = fftfreq(muestras, delta)
# Gráfico de la señal
plt.subplot(2,1,2)
plt.vlines(variableX, 0, np.abs(variableY))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud (Db)')
plt.show()
