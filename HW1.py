import numpy as np
import time
import matplotlib.pyplot as plt

dt=0.05
df=0.05

def recSTFT(x, t, f, B):
    Q = round(B/dt)
    N = round(1/(dt*df))
    stft = np.zeros((len(f), len(t)-(2*Q)), dtype=np.complex128)
    
    x1 = np.zeros(N+1)
    for n in range(Q,len(t)-Q):
        x1[:2*Q+1] = x[n-Q:n+Q+1]
        x1_fft = np.fft.fft(x1)
        stft[0:100,n-Q] = x1_fft[-100:]*np.exp(1j*2*np.pi*(Q-n)*t[n])*dt
        stft[100:201,n-Q] = x1_fft[:101]*np.exp(1j*2*np.pi*(Q-n)*t[n])*dt
        
    return stft


def main():
    t1=np.arange(0, 10, dt)
    t2=np.arange(10,20,dt)
    t3=np.arange(20,30+dt,dt)
    t=np.arange(0,30+dt,dt)
    f=np.arange(-5,5+df,df)
    x=np.concatenate((np.cos(2*np.pi*t1),np.cos(6*np.pi*t2),np.cos(4*np.pi*t3)),axis=0)
    B=1
    
    start_time = time.time()
    stft = recSTFT(x, t, f, B)
    end_time = time.time()
    total_time = end_time - start_time
    print('Total time: ', total_time)

    C = 400000
    stft = np.abs(stft) / np.max(np.abs(stft)) * C
    plt.imshow(stft, cmap='gray', origin='lower')
    plt.xlabel('Time (Sec)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

if __name__ == '__main__':
    main()