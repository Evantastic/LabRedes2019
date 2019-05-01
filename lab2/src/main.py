from scipy.io.wavfile import read, write
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import spectrogram, lfilter, lfilter_zi, butter, freqz
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


def pretty_firpass(lowcut, highcut, rate, numtaps):
    fir = firpass(lowcut, highcut, rate, numtaps)
    xAxis = np.arange(-len(fir)//2, len(fir)//2)
    return xAxis, fir

def filter(data, lowcut, highcut, rate, numtaps):
    fir = firpass(lowcut, highcut, rate, numtaps)
    y = convolve(ifft(fir), data)
    return y

# Analisis señal original
rate, amplitudeTime = read("../resources/audio/handel.wav")
amplitudeFrequency = fft(amplitudeTime)
frequency = fftfreq(len(amplitudeTime), 1 / rate)
f,t,spectrum = spectrogram(amplitudeTime, rate)

# Filtro FIR pasa banda entre dos peaks
w = [None] * 6
h = [None] * 6
filteredFrequencies = [None] * 6
fs = [None] * 6
ts = [None] * 6
spectrums = [None] * 6
for i in range(2):
    j = 2 + i
    k = 4 + i
    w[i], h[i] = pretty_firpass(280, 4000, rate, 2*i + 3)
    w[j], h[j] = pretty_firpass(280, 1670, rate, 2*i + 3)
    w[k], h[k] = pretty_firpass(0, 1670, rate, 2*i + 3)
    filteredFrequencies[i] = np.asarray(filter(amplitudeTime, 280, 4000, rate, 2*i + 3).real, dtype=np.int16)
    filteredFrequencies[j] = np.asarray(filter(amplitudeTime, 280, 1670, rate, 2*i + 3).real, dtype=np.int16)
    filteredFrequencies[k] = np.asarray(filter(amplitudeTime, 0, 1670, rate, 2*i + 3).real, dtype=np.int16)
    fs[i], ts[i], spectrums[i] = spectrogram(filteredFrequencies[i], rate)
    fs[j], ts[j], spectrums[j] = spectrogram(filteredFrequencies[j], rate)
    fs[k], ts[k], spectrums[k] = spectrogram(filteredFrequencies[k], rate)

# Guardar archivos
for i in range(2):
    j = 2 + i
    k = 4 + i
    write("../resources/audio/handelHighNumtaps%d.wav"%(5*(i + 1)), rate, filteredFrequencies[i])
    write("../resources/audio/handelLowNumtaps%d.wav"%(5*(i + 1)), rate, filteredFrequencies[k])
    write("../resources/audio/handelBandNumtaps%d.wav"%(5*(i + 1)), rate, filteredFrequencies[j])

# Errores
for i in range(2):
    j = 2 + i
    k = 4 + i
    orden = 2*i + 3
    ei = ((filteredFrequencies[i] - amplitudeTime) ** 2).mean()
    ej = ((filteredFrequencies[j] - amplitudeTime) ** 2).mean()
    ek = ((filteredFrequencies[k] - amplitudeTime) ** 2).mean()
    print("Error High Pass con Orden %d: %f"%(orden, ei))
    print("Error Band Pass con Orden %d: %f"%(orden, ej))
    print("Error Low Pass con Orden %d: %f"%(orden, ek))

# Primer espectrograma
plt.figure(1)
plt.pcolormesh(t,f,np.log10(spectrum))
plt.title("Espectrograma original")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.colorbar().set_label("Amplitud [Db]")

# Grafico de frequencias filtradas
plt.figure(3)
plt.clf()
for i in range(2):
    j = 2 + i
    k = 4 + i
    plt.plot(w[i], h[i], label="High Pass orden = %d"%(5*(i + 1)))
    plt.plot(w[j], h[j], label="Band Pass orden = %d"%(5*(i + 1)))
    plt.plot(w[k], h[k], label="Low Pass orden = %d"%(5*(i + 1)))
plt.grid(True)
plt.legend()
plt.xlabel("Frequencia [Hz]")
plt.ylabel("Amplitud")
plt.title("Señales filtradas")

# Grafico del espectrograma
for i in range(2):
    j = 2 + i
    k = 4 + i
    plt.figure(4 + i * 2)
    plt.pcolormesh(ts[i],fs[i],np.log10(spectrums[i]))
    plt.title("Espectrograma High Pass Orden = %d"%(5*(i + 1)))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.colorbar().set_label("Amplitud [Db]")
    plt.figure(5 + i * 2)
    plt.pcolormesh(ts[k],fs[k],np.log10(spectrums[k]))
    plt.title("Espectrograma Low Pass Orden = %d"%(5*(i + 1)))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.colorbar().set_label("Amplitud [Db]")
    plt.figure(6 + i * 2)
    plt.pcolormesh(ts[j],fs[j],np.log10(spectrums[j]))
    plt.title("Espectrograma Band Pass Orden = %d"%(5*(i + 1)))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Frecuencia [Hz]")
    plt.colorbar().set_label("Amplitud [Db]")

plt.show()
