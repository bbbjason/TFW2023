import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time

def hht(x, t, thr):
    y = x.copy()
    n = 1
    k = 1
    key = 0
    keys = 0
    length = len(y)
    c = np.zeros((10, length))

    while n < 10 and keys == 0:
        k = 1
        key = 0

        while k < 30 and key == 0:
            idmax = []
            idmin = []
            for i in range(1, length-1):
                if y[i] >= y[i-1] and y[i] >= y[i+1]:
                    idmax.append(i)
                elif y[i] <= y[i-1] and y[i] <= y[i+1]:
                    idmin.append(i)

            spmax = CubicSpline(t[idmax], y[idmax])(t)
            spmin = CubicSpline(t[idmin], y[idmin])(t)

            z = (spmax + spmin) / 2
            h = y - z

            key = 1
            hidmax = []
            hidmin = []
            for i in range(1, length-1):
                if h[i] >= h[i-1] and h[i] >= h[i+1]:
                    hidmax.append(i)
                elif h[i] <= h[i-1] and h[i] <= h[i+1]:
                    hidmin.append(i)

            u1 = CubicSpline(t[hidmax], h[hidmax])(t)
            u0 = CubicSpline(t[hidmin], h[hidmin])(t)

            for i in hidmax:
                if h[i] <= 0:
                    key = 0
                    break

            for i in hidmin:
                if h[i] >= 0:
                    key = 0
                    break

            sum_values = np.abs(u1 + u0)
            for i in range(length):
                if sum_values[i] >= thr:
                    key = 0
                    break

            if not (key == 0 and k < 30):
                c[n-1, :] = h
            else:
                y = h
                k += 1

        x0 = x.copy()
        for i in range(n):
            x0 -= c[i, :]

        iter_count = 0
        for i in range(1, length-1):
            if x0[i] >= x0[i-1] and x0[i] >= x0[i+1]:
                iter_count += 1
            elif x0[i] <= x0[i-1] and x0[i] <= x0[i+1]:
                iter_count += 1

        keys = 1
        if iter_count > 2:
            keys = 0

        if not (n < 10 and keys == 0):
            c[n, :] = x0

        y = x0
        n += 1

    return c[:n, :]

t = np.arange(0, 10, 0.01)
x = 0.2*t + np.cos(2*np.pi*t) + 0.4*np.cos(10*np.pi*t)
thr = 0.2

start_time = time.time()
y = hht(x, t, thr)
end_time = time.time()
print("Time taken: ", end_time - start_time)

plt.figure(1)
for i in range(y.shape[0]):
    plt.plot(t, y[i, :])
plt.axis([0, 10, -2, 2])
plt.show()

