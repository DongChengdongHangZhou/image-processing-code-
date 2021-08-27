import csv
import torch
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
from azimuth_integral import GetPSD1D

def write_csv():
    f = open('fake_fingerprint.csv','w',newline='')
    f_mean = open('mean_fake_fingerprint.csv','w',newline='')
    writer1 = csv.writer(f)
    writer2 = csv.writer(f_mean)
    sum = np.zeros((128,))
    for i in range(7400):
        try:
            print(i)
            dir = 'trainB/'+str(i)+'_B.tiff'
            spec = tiff.imread(dir)
            spec = 0.5*spec + 0.5
            spec = np.exp(spec*16)-1
            res = GetPSD1D(spec)
            res = res/res[0]
            sum = sum + res
            writer1.writerow(res)
        except:
            print('trainB/'+str(i)+'_B.tiff')

    mean = sum/7400
    writer2.writerow(mean)

def read_csv():
    with open('mean_results.csv','r')as f:
        f_csv = csv.reader(f)
        for row1 in f_csv:
            row1 = [float(i) for i in row1]
            print(row1)

    with open('mean_fake_fingerprint.csv','r')as f:
        f_csv = csv.reader(f)
        for row2 in f_csv:
            row2 = [float(i) for i in row2]
            print(row2)

    with open('mean_targetData.csv','r')as f:
        f_csv = csv.reader(f)
        for row3 in f_csv:
            row3 = [float(i) for i in row3]
            print(row3)
    plt.plot(row1,'red')
    plt.plot(row2,'blue')
    plt.plot(row3,'green')
    plt.show()

def draw_spec_all():
    data_results = pd.read_csv('./results.csv',header=None)
    data_results = np.array(data_results)
    data_fake_fingerprint = pd.read_csv('./fake_fingerprint.csv',header=None)
    data_fake_fingerprint = np.array(data_fake_fingerprint)
    data_targetData = pd.read_csv('./targetData.csv',header=None)
    data_targetData = np.array(data_targetData)
    for i in range(200):
        row1,row2,row3 = data_results[i], data_fake_fingerprint[i],data_targetData[i]
        row1 = [float(i) for i in row1]
        row2 = [float(i) for i in row2]
        row3 = [float(i) for i in row3]
        plt.plot(row1,'red')
        plt.plot(row2,'blue')
        plt.plot(row3,'green')
    plt.show()    

def crossEntropyLossValue(tensor1,tensor2):

    '''
    you must rewrite your own crossEntropyLoss since
    the pytorch version of crossEntropyLoss is
    (p(x)*log(q(x))).sum()
    but the crossEntropyLoss applied in this paper is
    (p(x)*log(q(x))+(1-p(x))*log(1-q(x))).sum()
    '''
    loss = tensor1*torch.log(tensor2)+(1-tensor1)*torch.log(1-tensor2)
    return loss

if __name__ == '__main__':
    draw_spec_all()



