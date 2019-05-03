# -*- coding: utf-8 -*-
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import spectrogram
from scipy.signal import firwin, convolve
import matplotlib.pyplot as plt
import numpy as np

def firpass(lowcut, highcut, rate, numtaps):
    if lowcut == 0:
        fir = firwin(numtaps, highcut, fs=rate)
    elif highcut == 4000:
        fir = firwin(numtaps, lowcut, fs=rate, pass_zero=False)
    else:
        fir = firwin(numtaps, [lowcut, highcut], fs=rate, pass_zero=False)
    return fir

def filter(data, lowcut, highcut, rate, numtaps):
    fir = firpass(lowcut, highcut, rate, numtaps)
    y = convolve(ifft(fir), data)
    return y

# Analisis señal original
rate, amplitudeTime = read("../resources/audio/handel.wav")
frequency = fftfreq(len(amplitudeTime), 1 / rate)
f,t,spectrum = spectrogram(amplitudeTime, rate)

# Filtro FIR pasa banda entre dos peaks
filteredFrequencies = [None] * 6
fs = [None] * 6
ts = [None] * 6
spectrums = [None] * 6
for i in range(2):
    j = 2 + i
    k = 4 + i
    numtaps = 2*i + 3
    filteredFrequencies[i] = np.asarray(filter(amplitudeTime, 280, 4000, rate, numtaps).real, dtype=np.int16)
    filteredFrequencies[j] = np.asarray(filter(amplitudeTime, 280, 1670, rate, numtaps).real, dtype=np.int16)
    filteredFrequencies[k] = np.asarray(filter(amplitudeTime, 0, 1670, rate, numtaps).real, dtype=np.int16)
    fs[i], ts[i], spectrums[i] = spectrogram(filteredFrequencies[i], rate)
    fs[j], ts[j], spectrums[j] = spectrogram(filteredFrequencies[j], rate)
    fs[k], ts[k], spectrums[k] = spectrogram(filteredFrequencies[k], rate)

# Guardar archivos
for i in range(2):
    j = 2 + i
    k = 4 + i
    numtaps = 2*i + 3
    write("../resources/audio/handelHighNumtaps%d.wav"%(numtaps), rate, filteredFrequencies[i])
    write("../resources/audio/handelLowNumtaps%d.wav"%(numtaps), rate, filteredFrequencies[k])
    write("../resources/audio/handelBandNumtaps%d.wav"%(numtaps), rate, filteredFrequencies[j])

# Errores
#for i in range(2):
#    j = 2 + i
#    k = 4 + i
#    numtaps = 2*i + 3
#    ei = ((filteredFrequencies[i] - amplitudeTime) ** 2).mean()
#    ej = ((filteredFrequencies[j] - amplitudeTime) ** 2).mean()
#    ek = ((filteredFrequencies[k] - amplitudeTime) ** 2).mean()
#    print("Error High Pass con Numtaps %d: %f"%(numtaps, ei))
#    print("Error Band Pass con Numtaps %d: %f"%(numtaps, ej))
#    print("Error Low Pass con Numtaps %d: %f"%(numtaps, ek))

# Primer espectrograma
plt.figure(1)
plt.pcolormesh(t,f,np.log10(spectrum))
plt.title("Espectrograma original")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.colorbar().set_label("Amplitud [Db]")

# Grafico del espectrograma
for i in range(2):
    j = 2 + i
    k = 4 + i
    numtaps = 2*i + 3
    plt.figure(2 + i * 3)
    plt.pcolormesh(ts[i],fs[i],np.log10(spectrums[i]))
    plt.title("Espectrograma High Pass Numtaps = %d"%(numtaps))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.colorbar().set_label("Amplitud [Db]")
    plt.figure(3 + i * 3)
    plt.pcolormesh(ts[k],fs[k],np.log10(spectrums[k]))
    plt.title("Espectrograma Low Pass Numtaps = %d"%(numtaps))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.colorbar().set_label("Amplitud [Db]")
    plt.figure(4 + i * 3)
    plt.pcolormesh(ts[j],fs[j],np.log10(spectrums[j]))
    plt.title("Espectrograma Band Pass Numtaps = %d"%(numtaps))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.colorbar().set_label("Amplitud [Db]")

plt.show()
