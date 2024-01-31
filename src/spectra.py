import numpy as np
import matplotlib.pyplot as plt


def spectrum(a, b, c, A, x):
    return A * (x ** (-a / c) + x ** (-b / c)) ** (-c)


# Lets plot some spectra
x = np.linspace(0, 3, 100)
y_6 = spectrum(4.5, -4.5, 2, 2, x)

# plt.plot(x,y_1,label='a=b=c=1')
# plt.plot(x,y_2,label='a=0')
# plt.plot(x,y_3,label='b=0')
# plt.plot(x,y_4,label='b=-1')
# plt.plot(x,y_5,label='b=-1,c=2')
plt.plot(x, y_6, label="a=4.5,b=-4.5,c=2")
plt.legend()
plt.show()
