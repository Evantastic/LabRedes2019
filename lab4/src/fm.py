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


def plot(time, signal, signal_am, signal_fm, index, carrier_frequency, figure):
    plt.figure(figure)
    ax1 = plt.subplot(311)
    ax1.get_xaxis().set_visible(False)
    plt.plot(time, signal)
    plt.title('Señal original')
    plt.ylabel('Amplitud')
    ax2 = plt.subplot(312,sharex = ax1)
    ax2.get_xaxis().set_visible(False)
    plt.plot(time, signal_am)
    plt.title('Señal con modulación AM con índice %f y frecuencia %d'%(index, carrier_frequency))
    plt.ylabel('Amplitud')
    ax3 = plt.subplot(313, sharex = ax1)
    plt.plot(time, signal_fm)
    plt.title('Señal con modulación FM con índice %f y frecuencia %d'%(index, carrier_frequency))
    plt.xlabel('Tiempo [x]')
    plt.ylabel('Amplitud')


time, signal, integral = initialize_data('../resources/handel.wav')
signal_am_100 = modulacion_am(signal, time, 1.0, 99100)
signal_fm_100 = modulacion_fm(integral, time, 1.0, 99100)
signal_am_015 = modulacion_am(signal, time, 0.15, 99100)
signal_fm_015 = modulacion_fm(integral, time, 0.15, 99100)
signal_am_125 = modulacion_am(signal, time, 1.25, 99100)
signal_fm_125 = modulacion_fm(integral, time, 1.25, 99100)
plot(time, signal, signal_am_100, signal_fm_100, 1.0, 99100, 1)
plot(time, signal, signal_am_015, signal_fm_015, 0.15, 99100, 2)
plot(time, signal, signal_am_125, signal_fm_125, 1.25, 99100, 3)
plt.show()
