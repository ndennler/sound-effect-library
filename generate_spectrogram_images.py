import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

for f in tqdm(os.listdir('spectrograms')):
    im = np.load(f'spectrograms/{f}')
    # a colormap and a normalization instance
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=im.min(), vmax=im.max())

    # map the normalized data to colors
    # image is now RGBA (512x512x4) 
    image = cmap(norm(im))
    # save the image
    plt.imsave(f'spectrogram_images/{f[:-4]}.jpg', image)

    # plt.imshow(im, interpolation='nearest')
    # plt.show(