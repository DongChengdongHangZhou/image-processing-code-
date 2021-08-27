import cv2
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('0.png',cv2.IMREAD_GRAYSCALE) #直接读为灰度图像
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude = np.abs(fshift) #取振幅
k = np.log(magnitude)
k /= k.max()/255
cv2.imwrite('ss.png',k)
phase = np.angle(fshift) #取相位
s1_real = magnitude*np.cos(phase) #取实部
s1_imag = magnitude*np.sin(phase) #取虚部
s2 = np.zeros(img.shape,dtype=complex)
s2.real = np.array(s1_real) #重新赋值给s2
s2.imag = np.array(s1_imag)

f2shift = np.fft.ifftshift(s2) #对新的进行逆变换
img_back = np.fft.ifft2(f2shift)
img_back = np.abs(img_back)

plt.subplot(221),plt.imshow(img,cmap='gray'),plt.title('original')
plt.subplot(222),plt.imshow(np.log(magnitude),cmap='gray'),plt.title('magnitude')
plt.subplot(223),plt.imshow(phase,cmap='gray'),plt.title('phase')
plt.subplot(224),plt.imshow(img_back,cmap='gray'),plt.title('another way')
plt.xticks([]),plt.yticks([])
plt.savefig('save.jpg')
plt.show()

