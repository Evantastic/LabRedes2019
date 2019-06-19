import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.io.wavfile import read


def modulacion_am(signal_array, time_array, index, carrier_frequency):
    aux = index*np.cos(2*np.pi*carrier_frequency*time_array)
    f = lambda x:x+1
    return list(map(f,signal_array))*aux


def modulacion_fm(integral_array, time_array, index, carrier_frequency):
    aux1 = 2*np.pi*carrier_frequency*time_array
    aux2 = 2*np.pi*index*integral_array
    suma = []
    for i in range(len(aux1)):
        suma.append(aux1[i] + aux2[i])
    return np.cos(np.array(suma))


def initialize_data(filename):
    rate, signal_array = read(filename)
    delta = 1/rate
    datalen = len(signal_array)
    time_array = np.linspace(0, (datalen - 1)*delta, datalen)
    integral_array = []
    for i in range(datalen):
        integral_array.append(simps(signal_array[:i+1], dx=delta))
    return time_array, signal_array, np.array(integral_array)


def plot(time, signal, signal_am, signal_fm, index, carrier_frequency, figure, carrier=[]):
    plt.figure(figure)
    ax1 = plt.subplot(311)
    ax1.get_xaxis().set_visible(False)
    if len(carrier) == 0:
        plt.plot(time, signal)
    else:
        plt.plot(time, signal, label='Original')
        plt.plot(time, carrier, label='Carrier')
        plt.legend()
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


def test(carrier_frequency):
    f = lambda z: np.cos(0.05*z*np.pi)
    x = np.linspace(0, 80, 40000)
    delta = 80.0/40000.0
    y = list(map(f,x))
    integrales = []
    for i in range(len(y)):
        integrales.append(simps(y[:i+1], dx=delta))
    integrales = np.array(integrales)
    ya = modulacion_am(y, x, 1.0, carrier_frequency)
    yf = modulacion_fm(integrales, x, 1.0, carrier_frequency)
    carrier = np.cos(2*np.pi*carrier_frequency*x)
    plot(x, y, ya, yf, 1.0, carrier_frequency, 1, carrier=carrier)
    plt.show()

    
def main():
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


test(2)
