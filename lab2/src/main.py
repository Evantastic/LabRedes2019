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

def pretty_bandpass(lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order)
    w, h = freqz(b, a, worN=2000)
    w, h = (rate * 0.5 / np.pi) * w, np.abs(h)
    return w, h


rate, amplitudeTime = read("../resources/audio/handel.wav")
amplitudeFrequency = np.abs(fft(amplitudeTime))
frequency = fftfreq(len(amplitudeTime), 1 / rate)
f,t,spectrum = spectrogram(amplitudeTime, rate)

# Filtro FIR pasa banda entre dos peaks
w1, h1 = pretty_bandpass(560, 1125, rate, order=1)
w2, h2 = pretty_bandpass(560, 1125, rate, order=2)
w3, h3 = pretty_bandpass(560, 1125, rate)

# Primer espectrograma
plt.figure(1)
plt.pcolormesh(t,f,np.log10(spectrum))
plt.title("Espectrograma")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.colorbar().set_label("Amplitud [Db]")

# Grafico de filtro fir
plt.figure(2)
plt.clf()
plt.plot(w1, h1, label="orden = 1")
plt.plot(w2, h2, label="orden = 2")
plt.plot(w3, h3, label="orden = 3")
plt.grid(True)
plt.legend()
plt.xlabel("Frequencia [Hz]")
plt.ylabel("Amplitud")
plt.title("Filtro FIR pasa banda")
plt.show()
