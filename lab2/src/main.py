from scipy.io.wavfile import read, write
from scipy.fftpack import fft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np

rate, amplitudeTime = read("../resources/audio/handel.wav")
frequencyTime = fft(amplitudeTime)
f,t,spectrum = spectrogram(amplitudeTime, rate)
plt.pcolormesh(t,f,np.log10(spectrum))
plt.title("Espectrograma")
plt.xlabel("Tiempo [s]")
plt.ylabel("Frecuencia [Hz]")
plt.colorbar().set_label("Amplitud [Db]")
plt.show()
