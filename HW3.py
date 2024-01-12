import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    t = np.arange(-9, 9, 0.0125)
    f = np.arange(-4, 4, 0.025)
    x = np.exp(1j * t**2 / 10 - 1j * 3 * t) * ((t >= -9) & (t <= 1)) + np.exp(1j * t**2 / 2 + 1j * 6 * t) * np.exp(-(t - 4)**2 / 10)

    start_time = time.time()
    y = wdf(x, t, f)
    end_time = time.time()
    total_time = end_time - start_time
    print('Total time: ', total_time)

    drawgraph(y, f, t)

def wdf(x, t, f):
    dt = t[1] - t[0]
    df = f[1] - f[0]
    N = round(1 / (2 * dt * df))
    n1 = round(t[0] / dt)
    n2 = round(t[-1] / dt)
    m1 = round(f[0] / df)
    m2 = round(f[-1] / df)
    m = np.mod(np.arange(m1, m2+1), N) + 1
    Lt = n2 - n1 + 1
    Lf = m2 - m1 + 1
    y = np.zeros((Lf, Lt), dtype=complex)
    for n in range(n1, n2+1):
        U = min(n2 - n, n - n1)
        Q = 2 * U + 1
        A = x[n - n1 + np.arange(-U, U+1)] * np.conj(x[n - n1 + np.arange(U, -U-1, -1)])
        A1 = np.fft.fft(A, N) * 2 * dt
        a1 = np.ceil(Q / N).astype(int)
        for a2 in range(2, a1 + 1):
            A1 += np.fft.fft(A[(a2 - 1) * N : min(a2 * N, Q)], N) * 2 * dt
        y[:, n - n1] = A1[m - 1] * np.exp(1j * 2 * np.pi / N * U * (m - 1))
    return y

def drawgraph(y, f, t):
    plt.imshow(np.abs(y) / np.max(np.abs(y)) * 400, extent=[t[0], t[-1], f[-1], f[0]], aspect='auto', cmap='gray')
    plt.gca().invert_yaxis()
    plt.xlabel('Time (Sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Wigner Distribution Function')
    plt.show()

main()