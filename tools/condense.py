import numpy as np
import skimage.io
from scipy import fftpack
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


# Show the results

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()


def process(img):
    rgb = img.mean(axis=(0, 1))
    return rgb

    # DEAD CODE NOW

    bw = rgb2gray(img)

    plt.imshow(bw)
    plt.show()

    fft = fftpack.fft2(bw)

    plt.figure()
    plot_spectrum(fftpack.fftshift(fft))
    plt.show()

    print(np.histogram(np.real(fft)))

    do_threshold = False
    if do_threshold:
        threshold = np.abs(fft).mean() * 5
        print(threshold)
        fft_thr = fft * (np.abs(fft) > threshold).astype(float)
    else:
        x, y = fft.shape
        assert x == y
        window = np.ones((x, y))
        keep = x // 10
        window[keep : x - keep, :] = 0
        window[:, keep : x - keep] = 0
        # plt.imshow(window) ; plt.show()
        fft_thr = fft * window

    plot_spectrum(fftpack.fftshift(fft_thr))
    plt.show()

    img_thr = np.real(fftpack.ifft2(fft_thr)) * np.sqrt(img.size)
    plt.imshow(img_thr)
    plt.show()


def process_many():
    for l in sys.stdin:
        filename = l.strip()
        img = skimage.io.imread(filename)
        print(filename + "\t" + "\t".join(map(str, process(img))))

# process_many() ; exit()

filename, = sys.argv[1:]
img = skimage.io.imread(filename)
print("\t".join(map(str, process(img))))
exit()

img = np.zeros((256, 256))
img += 128 + 128 * np.sin(np.arange(256) / 256 * np.pi * 20)[:, np.newaxis]
img += np.random.normal(0, 20, size=img.shape)
plt.imshow(img)
plt.show()
print(process(img))

