import matplotlib.pyplot as plt
import numpy as np
from random import choice


def modulador(bit, frequency, rate, puntos, t = 0):
    ts = np.linspace(t, t+rate,puntos)
    t+=rate
    return 50*bit*np.cos(frequency*ts)
    

bitQuantity = 15
frequency = 100
puntos = frequency/2
rate = 0.1
bitsLen = int(puntos*bitQuantity)

bitString = ''.join(choice(['0', '1']) for _ in range(bitQuantity))

array = []
for c in bitString:
    array.extend(modulador(int(c), frequency*2*np.pi, rate, puntos))


# A continuación viene el ploteo
time = np.linspace(0, bitQuantity*rate, puntos*bitQuantity)
plt.plot(time, array)
plt.title(bitString)
plt.show()
