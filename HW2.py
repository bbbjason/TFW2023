import scipy.io.wavfile as wav
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def Gabor(x, tau, t, f, sgm):
    dtau = tau[1]-tau[0]
    dt = t[1]-t[0]
    df = f[1]-f[0]
    T = len(tau)
    N = round(1/(dtau*df))
    F = len(f)
    C = len(t)
    Q = round(1.9143/(math.sqrt(sgm)*dtau))
    S = round(dt/dtau)

    X = np.zeros((f[-1],C), dtype=np.complex128)
    for n in (range(0, C)):
        x1 = np.zeros(N)
        x_pad = np.pad(x, (0, Q), 'constant')
        q = np.arange(0, 2*Q+1)
        x1[:2*Q+1] = np.multiply(x_pad[(n*S+q-Q)], np.exp(-sgm*math.pi*((q-Q)*dtau)**2))
        # x1[:2*Q+1] = x_pad[(n*S+q-Q)]
        X1 = np.fft.fft(x1)
        X[f[0]:f[-1], n] = X1[f[0]:f[-1]]
    return X

def main():
    rt, wavsignal = wav.read('Chord.wav')
    x=wavsignal[:,1]
    dtau = 1/44100
    dt = 0.01
    df= 1
    tau = np.arange(0, 1.6+dtau, dtau)
    t = np.arange(0, 1.6+dt, dt)
    f = np.arange(20, 1000+df, df)
    sgm= 200
    
    start_time = time.time()
    y = Gabor(x, tau, t, f, sgm)
    end_time = time.time()
    total_time = end_time - start_time
    print('Total time: ', total_time)
    
    C = 1000000
    y = np.abs(y) / np.max(np.abs(y)) * C
    plt.imshow(y, cmap='gray', origin='lower', aspect=0.1, extent=[0, 160, 20, 1000])
    plt.xlabel('Time (0.01*Sec)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

if __name__ == '__main__':
    main()