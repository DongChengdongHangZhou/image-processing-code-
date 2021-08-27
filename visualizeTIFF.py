import cv2
import os
from tifffile.tifffile import imwrite
import torch
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import math



img = tiff.imread('./15.tiff')
img = np.exp((0.5*img + 0.5))
plt.subplot(111),plt.imshow(np.log(img),cmap='gray'),plt.title('magnitude')
plt.savefig('save.jpg')
plt.show()



