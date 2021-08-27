import cv2
import os
from numpy.lib.shape_base import kron
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('0.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('save.jpg',cv2.IMREAD_GRAYSCALE)
k = np.zeros((256,256),dtype=np.float32)

for i in range(256):
    for j in range(256):
        if img1[i][j]>img2[i][j]:
            k[i][j] = img1[i][j]-img2[i][j]+128
        if img1[i][j]<=img2[i][j]:
            k[i][j] = 128-(img2[i][j]-img1[i][j])

plt.subplot(111),plt.imshow(k[50:120,50:120],cmap='gray'),plt.title('magnitude')
plt.show()
