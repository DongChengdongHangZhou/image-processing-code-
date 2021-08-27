from scipy import ndimage
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2


def butterworth_Filter_LowPass(shape, rank, radius):
    """ butterworth filter genreator

    H(u, v) = 1 / (1 + (D(u, v) / radius)^(2 * rank))
    Args:
        shape:  滤波器的形状
        rank:   滤波器阶数
        radius: 滤波器半径
    """
    # 中心位置
    h = w = shape
    cx, cy = int(w / 2), int(h / 2)
    # 计算以中心为原点坐标分量
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    # 每个点到中心的距离
    dis = np.sqrt(u * u + v * v)
    filt = 1 / (1 + np.power(dis / radius, 2 * rank))
    return filt

def butterworth_Filter_HighPass(shape, rank, radius):
    h = w = shape
    cx, cy = int(w / 2), int(h / 2)
    # 计算以中心为原点坐标分量
    u = np.array([[x - cx for x in range(w)] for i in range(h)], dtype=np.float32)
    v = np.array([[y - cy for y in range(h)] for i in range(w)], dtype=np.float32).T
    # 每个点到中心的距离
    dis = np.sqrt(u * u + v * v)
    filt = 1 - 1 / (1 + np.power(dis / radius, 2 * rank))
    return filt

LowPass = butterworth_Filter_LowPass(256,2,45)
HighPass = butterworth_Filter_HighPass(256,2,45)

def GetPSD1D(psd2D):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2
    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)
    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # NOTE: this will miss power contributions in 'corners' r>wc
    psd1D = ndimage.sum(psd2D, r, index=np.arange(0, wc))
    return psd1D

def spectrum_adjust(mat,energy_list1,energy_list2):
    w  = mat.shape[0]
    wc = w//2
    Y, X = np.ogrid[0:w, 0:w]
    r = np.hypot(X - wc, Y - wc).astype(np.int)
    for i in range(w):
        for j in range(w):
            value = r[i][j]
            if value<wc:
                mat[i][j] = mat[i][j]*energy_list1[value]/energy_list2[value]
    return mat

def process(spec,img_synth):
    real_style_mag = 0.5*spec + 0.5
    f = np.fft.fft2(img_synth)
    fshift = np.fft.fftshift(f)
    phase = np.angle(fshift) #取相位
    magnitude = np.abs(fshift) #取振幅
    coef = np.log(magnitude).max()
    real_style_mag = np.exp(real_style_mag*coef)-1
    result1 = GetPSD1D(real_style_mag)
    result2 = GetPSD1D(magnitude)
    spec2 = spectrum_adjust(magnitude,result1,result2)
    freq = real_style_mag*HighPass + spec2*LowPass
    s1_real = freq*np.cos(phase) #取实部
    s1_imag = freq*np.sin(phase) #取虚部
    s2 = np.zeros(img_synth.shape,dtype=complex)
    s2.real = np.array(s1_real) #重新赋值给s2
    s2.imag = np.array(s1_imag)
    f2shift = np.fft.ifftshift(s2) #对新的进行逆变换
    img_back = np.fft.ifft2(f2shift)
    img_back = np.abs(img_back)
    return img_back


if __name__=="__main__":
    spec = tiff.imread('./results/0_B_fake_B.tiff')  #spec is artifact-decreased fourier 
    img_synth = cv2.imread('D:/Graduation_Dissertation/fake_fingerprint/0_A_fake_A.png',cv2.IMREAD_GRAYSCALE) #img_synth: synthetic image
    img_back = process(spec,img_synth)
    plt.subplot(121),plt.imshow(img_synth,cmap='gray'),plt.title('img_synth')
    plt.subplot(122),plt.imshow(img_back,cmap='gray'),plt.title('final_image')
    plt.show()


