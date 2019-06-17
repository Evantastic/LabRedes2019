import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.io.wavfile import read


def modulacion_am(signal_array, time_array, index, carrier_frequency):
    aux = index*np.cos(2*np.pi*carrier_frequency*time_array)
    return signal_array*aux


def modulacion_fm(integral_array, time_array, index, carrier_frequency):
    aux1 = 2*np.pi*carrier_frequency*time_array
    aux2 = index*integral_array
    return np.cos(aux1+aux2)


def initialize_data(filename):
    rate, signal_array = read(filename)
    delta = 1/rate
    datalen = len(signal_array)
    time_array = np.linspace(0, (datalen - 1)*delta, datalen)
    integral_array = []
    for i in range(datalen):
        integral_array.append(simps(signal_array[:i+1], dx=delta))
    return time_array, signal_array, np.array(integral_array)


time, signal, integral = initialize_data('../resources/handel.wav')
signal_am = modulacion_am(signal, time, 1.0, 99100)
signal_fm = modulacion_fm(integral, time, 1.0, 99100)

error = 0.0
for i in range(len(signal)):
    error += np.abs(signal[i] - signal_am[i])
error = error/len(signal)

print('Error: %d'%(error))

plt.figure(1)
plt.subplot(311)
plt.plot(time, signal)
plt.subplot(312)
plt.plot(time, signal_am)
plt.subplot(313)
plt.plot(time, signal_fm)
plt.show()
