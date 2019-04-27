from scipy.io.wavfile import read, write
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import spectrogram, lfilter, lfilter_zi, butter, freqz
import matplotlib.pyplot as plt
import numpy as np

def butter_bandpass(lowcut, highcut, rate, order):
    nyq = 0.5 * rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def pretty_bandpass(lowcut, highcut, rate, order, maximum=1):
    b, a = butter_bandpass(lowcut, highcut, rate, order)
    w, h = freqz(b, a, worN=2000)
    w, h = (rate * 0.5 / np.pi) * w, np.abs(h * maximum)
    return w, h

def filter(data, lowcut, highcut, rate, order):
    b, a = butter_bandpass(lowcut, highcut, rate, order)
    y = lfilter(b, a, data)
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
inverseAmplitudeTime = [None] * 6
fs = [None] * 6
ts = [None] * 6
spectrums = [None] * 6
for i in range(2):
    j = 2 + i
    k = 4 + i
    w[i], h[i] = pretty_bandpass(280, 4000, rate, 5 * (i + 1), maximum=2.400e7)
    w[j], h[j] = pretty_bandpass(280, 1670, rate, 5 * (i + 1), maximum=2.400e7)
    w[k], h[k] = pretty_bandpass(1, 1670, rate, 5 * (i + 1), maximum=2.400e7)
    filteredFrequencies[i] = filter(amplitudeFrequency, 280, 4000, rate, 5*(i + 1))
    filteredFrequencies[j] = filter(amplitudeFrequency, 280, 1670, rate, 5*(i + 1))
    filteredFrequencies[k] = filter(amplitudeFrequency, 1, 1670, rate, 5*(i + 1))
    inverseAmplitudeTime[i] = np.asarray(ifft(filteredFrequencies[i]).real, dtype=np.int16)
    inverseAmplitudeTime[j] = np.asarray(ifft(filteredFrequencies[j]).real, dtype=np.int16)
    inverseAmplitudeTime[k] = np.asarray(ifft(filteredFrequencies[k]).real, dtype=np.int16)
    fs[i], ts[i], spectrums[i] = spectrogram(inverseAmplitudeTime[i], rate)
    fs[j], ts[j], spectrums[j] = spectrogram(inverseAmplitudeTime[j], rate)
    fs[k], ts[k], spectrums[k] = spectrogram(inverseAmplitudeTime[k], rate)

# Guardar archivos
for i in range(2):
    j = 2 + i
    k = 4 + i
    write("../resources/audio/handelHighOrden%d.wav"%(5*(i + 1)), rate, inverseAmplitudeTime[i])
    write("../resources/audio/handelLowOrden%d.wav"%(5*(i + 1)), rate, inverseAmplitudeTime[k])
    write("../resources/audio/handelBandOrden%d.wav"%(5*(i + 1)), rate, inverseAmplitudeTime[j])

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
plt.plot(frequency, np.abs(amplitudeFrequency), label="original")
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
