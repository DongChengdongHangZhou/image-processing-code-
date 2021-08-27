import cv2
import os
import tifffile as tiff
import numpy as np

rootdir = 'D:/Graduation_Dissertation/fake_fingerprint'
write_dir = './trainB/'


for i in range(7400):
    try:
        dir = rootdir + '/' + str(i)+'_A_fake_A.png'
        print(i)
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)  # 取振幅
        k = np.log(magnitude+1)
        k /= k.max() 
        k = (k-0.5)*2
        tiff.imsave(write_dir+str(i)+'_B.tiff', k)
    except:
        print('error:',i)


