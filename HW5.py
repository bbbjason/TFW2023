from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import zoom

def wavedbc10(x):
    g = np.array([0.0033, -0.0126, -0.0062, 0.0776, -0.0322, -0.2423, 0.1384, 0.7243, 0.6038, 0.1601])
    h = np.array([0.1601, -0.6038, 0.7243, -0.1384, -0.2423, 0.0322, 0.0776, 0.0062, -0.0126, -0.0033])

    xg = convolve2d(x, g[:, None], mode='same')
    v1L = xg[:, ::2]
    xh = convolve2d(x, h[:, None], mode='same')
    v1H = xh[:, ::2]

    v1Lg = convolve2d(v1L, g[None, :], mode='same')
    x1L = v1Lg[::2, :]
    v1Lh = convolve2d(v1L, h[None, :], mode='same')
    x1H1 = v1Lh[::2, :]

    v1Hg = convolve2d(v1H, g[None, :], mode='same')
    x1H2 = v1Hg[::2, :]
    v1Hh = convolve2d(v1H, h[None, :], mode='same')
    x1H3 = v1Hh[::2, :]

    y = np.concatenate((x1L, x1H1, x1H2, x1H3), axis=1)
    return y


def iwavedbc10(x1L, x1H1, x1H2, x1H3):
    g1 = np.array([0.1601, -0.6038, 0.7243, -0.1384, -0.2423, 0.0322, 0.0776, 0.0062, -0.0126, -0.0033])
    h1 = np.array([-0.0033, -0.0126, 0.0062, 0.0776, 0.0322, -0.2423, -0.1384, 0.7243, -0.6038, 0.1601])

    def upsample_and_convolve(subband, filter):
        upsampled = np.zeros((subband.shape[0], subband.shape[1] * 2))
        upsampled[:, ::2] = subband
        return convolve2d(upsampled, filter.reshape(1, -1), mode='full')

    x0 = upsample_and_convolve(x1L, g1) + upsample_and_convolve(x1H1, h1)
    x1 = upsample_and_convolve(x1H2, g1) + upsample_and_convolve(x1H3, h1)

    def transpose_upsample_and_convolve(subband, filter):
        upsampled = np.zeros((subband.T.shape[0], subband.T.shape[1] * 2))
        upsampled[:, ::2] = subband.T
        return convolve2d(upsampled, filter[:, None], mode='full')

    newx = transpose_upsample_and_convolve(x0, g1) + transpose_upsample_and_convolve(x1, h1)

    cropped_newx = newx[5:-5, 5:-5]
    cropped_newx = cropped_newx.T[5:-5, 5:-5].T

    return cropped_newx

x = np.array(Image.open('./cat.jpg').convert('L'), dtype=np.float64)
y = wavedbc10(x)

L = y.shape[1] // 4
x1L, x1H1, x1H2, x1H3 = y[:, :L], y[:, L:2*L], y[:, 2*L:3*L], y[:, 3*L:]

plt.figure(1)
plt.imshow(x, cmap='gray')
plt.title('Original Image')
plt.savefig('original_image.png')

plt.figure(2)
plt.subplot(2, 2, 1)
plt.imshow(x1L / np.sqrt(2), cmap='gray')
plt.title('x1L')

plt.subplot(2, 2, 3)
plt.imshow(x1H1 * 5, cmap='gray')
plt.title('x1H1')

plt.subplot(2, 2, 2)
plt.imshow(x1H2 * 5, cmap='gray')
plt.title('x1H2')

plt.subplot(2, 2, 4)
plt.imshow(x1H3 * 10, cmap='gray')
plt.title('x1H3')
plt.savefig('wave_transformed.png')

z = iwavedbc10(x1L, x1H1, x1H2, x1H3)
print(z.shape)

plt.figure(3)
plt.imshow(z, cmap='gray')
plt.title('Reconstructed Image')
plt.savefig('reconstructed_image.png')

