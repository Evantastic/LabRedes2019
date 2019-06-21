import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.io.wavfile import read, write
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, filtfilt


# Function that modulates a discretized signal represented by time_array and signal_array.
# The modulation takes into consideration an index and a carrier to compute the new signal.
# For better visualization of the functionatity of the function, and offset can be given.
# This offset has to be the peak to peak value.
# It returns an array
def modulacion_am(signal_array, time_array, index, carrier_frequency, offset=0):
    aux = index*np.cos(2*np.pi*carrier_frequency*time_array)
    f = lambda x:x+offset
    return list(map(f,signal_array))*aux


# Function that demodulates a discritized signal represented by time_array and signal_array.
# The modulation modulates the already modulated signal given an index and a carrier_frequency.
# For the demudulation, it applies a lowpass filter that ignores frequencies that surpass the cut limit.
# It also takes into consideration the rate at which te original signal is sampled
# It returns an array
def demodulacion_am(signal_am_array, time_array, index, carrier_frequency, cut, rate):
    aux = 2*modulacion_am(signal_am_array, time_array, index, carrier_frequency)
    b, a = butter(3, cut , 'lowpass', fs=rate)
    filtered = filtfilt(b, a, aux)
    return filtered


# Function that modulates a discretized signal represented by time_array and integral of the signal.
# In general, the time_array and integral_array needs to follow this rule:
# Integral_array[i] is the integral of the signal from 0 to time_array[i]
# The modulation takes into consideration an index and a carrier to compute the new signal.
# For better visualization of the functionatity of the function, and offset can be given.
# This offset has to be the peak to peak value.
# It returns an array
def modulacion_fm(integral_array, time_array, index, carrier_frequency):
    aux1 = 2*np.pi*carrier_frequency*time_array
    aux2 = 2*np.pi*index*integral_array
    suma = []
    for i in range(len(aux1)):
        suma.append(aux1[i] + aux2[i])
    return np.cos(np.array(suma))


# Function that, given a filename, reads its contents and obtain the necessary information about the signal.
# It return the time and amplitude of the signal, the integral of the signal, and the rate at which it was sampled.
# For the calculation of the integral, the simps function is used.
# This is because the simps function calculates the integral of a discretized signal, assuming that the original 
# signal the x axis is equispaced. It calculates the integral of the whole array, thats the reason a subarray is given.
def initialize_data(filename):
    rate, signal_array = read(filename)
    delta = 1/rate
    datalen = len(signal_array)
    time_array = np.linspace(0, (datalen - 1)*delta, datalen)
    integral_array = []
    for i in range(datalen):
        integral_array.append(simps(signal_array[:i+1], dx=delta))
    return time_array, signal_array, np.array(integral_array), rate


# Function that plots in a 'prettier' way a signal, its am modulation, its fm modulation in the frequency domain
# If the modulation flag is set, that means the fm modulation is the am modulated signal that was modulated once more.
def plot_frequency(frequency, signal, signal_am, signal_fm, index, carrier_frequency, figure, demodulacion=False):
    plt.figure(figure)
    ax1 = plt.subplot(311)
    ax1.get_xaxis().set_visible(False)
    plt.vlines(frequency, 0, np.abs(signal))
    plt.title('Senal original')
    plt.ylabel('Amplitud')
    plt.xlabel('Tiempo [s]')
    ax2 = plt.subplot(312, sharex = ax1)
    ax2.get_xaxis().set_visible(False)
    plt.vlines(frequency,0, signal_am)
    plt.title('Senal con modulacion AM con indice %f y frecuencia %d'%(index, carrier_frequency))
    plt.ylabel('Amplitud')
    ax3 = plt.subplot(313, sharex = ax1)
    plt.vlines(frequency,0, signal_fm)
    if demodulacion:
        plt.title('Senal moduladada con modulacion AM con indice %f y frecuencia %d'%(index, carrier_frequency))
    else:
        plt.title('Senal con modulacion FM con indice %f y frecuencia %d'%(index, carrier_frequency))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')


# Function that plots in a 'prettier' way a signal, its am modulation, its fm modulation in the time domain
# If the carrier exists, that means that the carrier wave was included for plotting
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
    plt.title('Senal original')
    plt.ylabel('Amplitud')
    ax2 = plt.subplot(312,sharex = ax1)
    ax2.get_xaxis().set_visible(False)
    plt.plot(time, signal_am)
    plt.title('Senal con modulacion AM con indice %f y frecuencia %d'%(index, carrier_frequency))
    plt.ylabel('Amplitud')
    ax3 = plt.subplot(313, sharex = ax1)
    plt.plot(time, signal_fm)
    plt.title('Senal con modulacion FM con indice %f y frecuencia %d'%(index, carrier_frequency))
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')


# Function that test the previous functions given a sinusoidal function.
# If the modulation is to be tested, a low carrier frequency and an offset of 2 is needed.
# Thats because the offset is the value peak to peak of the tinusoidal function, and a low 
# carrier frequency allows to see clearly the plot.
# If the demodulation is to be teste, a high carrier frequency is needed (at least 10) and an
# offses of 0 is needed.
# A high frequency allows por a better aproximation of the demodulated rignal and the offset allows that
# the original signal and the demodulated one superimpose one another in the plot
def test(carrier_frequency, offset):
    f = lambda z: np.cos(0.05*z*np.pi)
    x = np.linspace(0, 80, 20000)
    delta = 80.0/20000.0
    y = list(map(f,x))
    integrales = []
    for i in range(len(y)):
        integrales.append(simps(y[:i+1], dx=delta))
    integrales = np.array(integrales)
    ya = modulacion_am(y, x, 1.0, carrier_frequency, offset=offset)
    yaa = modulacion_am(ya, x, 1.0, carrier_frequency, offset=offset*2)
    yf = modulacion_fm(integrales, x, 1.0, carrier_frequency)
    ydm = demodulacion_am(ya, x, 1.0, carrier_frequency, 9.0, 40000)
    fo = fft(y)
    foa = fft(ya)
    foaa = fft(yaa)
    fof = fft(yf)
    fr = fftfreq(len(y), delta)
    carrier = np.cos(2*np.pi*carrier_frequency*x)
    plot_signal(x, y, ya, yf, 1.0, carrier_frequency, 1)
    plot_frequency(fr, np.abs(fo), np.abs(foa), np.abs(fof), 1.0, carrier_frequency, 2)
    plot_frequency(fr, np.abs(fo), np.abs(foa), np.abs(foaa), 1.0, carrier_frequency, 3)
    plt.figure(4)
    plt.plot(x, ydm, label='Demodulada')
    plt.plot(x, y, label='Original')
    plt.title('Comparacion resultados con senal original')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.show()


# Funciont that runs the main objective of the project
def main():
    time, signal, integral, rate = initialize_data('../resources/handel.wav')
    signal_am_100 = modulacion_am(signal, time, 1.0, 99100)
    signal_fm_100 = modulacion_fm(integral, time, 1.0, 99100)
    signal_am_015 = modulacion_am(signal, time, 0.15, 99100)
    signal_fm_015 = modulacion_fm(integral, time, 0.15, 99100)
    signal_am_125 = modulacion_am(signal, time, 1.25, 99100)
    signal_fm_125 = modulacion_fm(integral, time, 1.25, 99100)
    signal_am_am = modulacion_am(signal_am_100, time, 1.0, 99100)
    signal_og_am = demodulacion_am(signal_am_100, time, 1.0, 99100, 1900.0, rate)
    write('../resources/handelRecreado.wav', rate, np.asarray(signal_og_am, dtype=np.int16))
    fourier = fft(signal)
    frequency = fftfreq(len(signal), 1.0/rate)
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
    plot_frequency(frequency, np.abs(fourier), np.abs(fourier_am_100), np.abs(fourier_fm_100), 1.0, 99100, 4)
    plot_frequency(frequency, np.abs(fourier), np.abs(fourier_am_015), np.abs(fourier_fm_015), 0.15, 99100, 5)
    plot_frequency(frequency, np.abs(fourier), np.abs(fourier_am_125), np.abs(fourier_fm_125), 1.25, 99100, 6)
    plot_frequency(frequency, np.abs(fourier), np.abs(fourier_am_100), np.abs(fourier_am_am), 1.0, 99100, 7, demodulacion=True)
    plt.show()

# Test para mostrar que la modulacion am y fm funcionan
#test(2,2)
# Test para mostrar que la demodulacion funciona
#test(10,0)
# Programa principal
main()
