import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.io.wavfile import read
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt




def modulacion_am(signal_array, time_array, index, carrier_frequency, test=False, offset=0):
    aux = index*np.cos(2*np.pi*carrier_frequency*time_array)
    f = lambda x:x+offset
    return list(map(f,signal_array))*aux


def demodulacion_am(signal_am_array, time_array, index, carrier_frequency, cut, rate):
    aux = 2*modulacion_am(signal_am_array, time_array, index, carrier_frequency)
    b, a = butter(3, cut , 'lowpass', fs=rate)
    filtered = filtfilt(b, a, aux)
    return filtered


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
    return time_array, signal_array, np.array(integral_array), delta


def plot_frequency(frequency, signal_am, signal_fm, index, carrier_frequency, figure, demodulacion=False):
    plt.figure(figure)
    ax1 = plt.subplot(211)
    ax1.get_xaxis().set_visible(False)
    plt.vlines(frequency,0, signal_am)
    plt.title('Señal con modulación AM con índice %f y frecuencia %d'%(index, carrier_frequency))
    plt.ylabel('Amplitud')
    ax3 = plt.subplot(212, sharex = ax1)
    plt.vlines(frequency,0, signal_fm)
    if demodulacion:
        plt.title('Señal con modulación AM con índice %f y frecuencia %d'%(index, carrier_frequency))
    else:
        plt.title('Señal con modulación FM con índice %f y frecuencia %d'%(index, carrier_frequency))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')


def plot_signal(time, signal, signal_am, signal_fm, index, carrier_frequency, figure, carrier=[]):
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
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')


def test(carrier_frequency, offset):
    f = lambda z: np.cos(0.05*z*np.pi)
    x = np.linspace(0, 80, 20000)
    delta = 80.0/20000.0
    y = list(map(f,x))
    integrales = []
    for i in range(len(y)):
        integrales.append(simps(y[:i+1], dx=delta))
    integrales = np.array(integrales)
    ya = modulacion_am(y, x, 1.0, carrier_frequency, test=True, offset=offset)
    yaa = modulacion_am(ya, x, 1.0, carrier_frequency, test=True, offset=offset*2)
    yf = modulacion_fm(integrales, x, 1.0, carrier_frequency)
    ydm = demodulacion_am(ya, x, 1.0, carrier_frequency, 9.0, 40000)
    fo = fft(y)
    foa = fft(ya)
    foaa = fft(yaa)
    fof = fft(yf)
    fr = fftfreq(len(y), delta)
    carrier = np.cos(2*np.pi*carrier_frequency*x)
    plot_signal(x, y, ya, yf, 1.0, carrier_frequency, 1)
    plt.figure(2)
    plt.vlines(fr, 0, np.abs(fo))
    plot_frequency(fr, np.abs(foa), np.abs(fof), 1.0, carrier_frequency, 3)
    plot_frequency(fr, np.abs(foa), np.abs(foaa), 1.0, carrier_frequency, 4)
    print(ydm)
    plt.figure(5)
    plt.plot(x, ydm, label='Demodulada')
    plt.plot(x, y, label='Original')
    plt.legend()
    plt.show()

    
def main():
    time, signal, integral, delta = initialize_data('../resources/handel.wav')
    signal_am_100 = modulacion_am(signal, time, 1.0, 99100)
    signal_fm_100 = modulacion_fm(integral, time, 1.0, 99100)
    signal_am_015 = modulacion_am(signal, time, 0.15, 99100)
    signal_fm_015 = modulacion_fm(integral, time, 0.15, 99100)
    signal_am_125 = modulacion_am(signal, time, 1.25, 99100)
    signal_fm_125 = modulacion_fm(integral, time, 1.25, 99100)
    signal_am_am = modulacion_am(signal_am_100, time, 1.0, 99100)
    fourier = fft(signal)
    frequency = fftfreq(len(signal), delta)
    fourier_am_100 = fft(signal_am_100)
    fourier_fm_100 = fft(signal_fm_100)
    fourier_am_015 = fft(signal_am_015)
    fourier_fm_015 = fft(signal_fm_015)
    fourier_am_125 = fft(signal_am_125)
    fourier_fm_125 = fft(signal_fm_125)
    fourier_am_am = fft(signal_am_am)
    plot_signal(time, signal, signal_am_100, signal_fm_100, 1.0, 99100, 1)
    plot_signal(time, signal, signal_am_015, signal_fm_015, 0.15, 99100, 2)
    plot_signal(time, signal, signal_am_125, signal_fm_125, 1.25, 99100, 3)
    plt.figure(4)
    plt.plot(frequency, fourier)
    plt.title('Señal original')
    plt.ylabel('Amplitud')
    plt.xlabel('Tiempo [s]')
    plot_frequency(frequency, fourier_am_100, fourier_fm_100, 1.0, 99100, 5)
    plot_frequency(frequency, fourier_am_015, fourier_fm_015, 0.15, 99100, 6)
    plot_frequency(frequency, fourier_am_125, fourier_fm_125, 1.25, 99100, 7)
    plot_frequency(frequency, fourier_am_100, fourier_am_am, 1.0, 99100, 8, demodulacion=True)
    plt.show()

# Test para mostrar que la modulacion am y fm funcionan
# test(2,2)
# Test para mostrar que la demodulacion funciona
# test(10,0)
# Programa principal
#main()
