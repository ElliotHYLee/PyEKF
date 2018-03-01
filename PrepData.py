import numpy as np
import pandas as pd

def readAttRaw():
    fName = 'Data/att_raw.txt'
    data = pd.read_csv(fName, sep=" ", header=None).as_matrix()
    gyro = data[:-1,0:3]
    acc = data[:-1,3:6]
    mag = data[:-1,6:9]
    return gyro, acc, mag

def readAttEst():
    fName = 'Data/att_dcm.txt'
    dcm = pd.read_csv(fName, sep=" ", header=None).as_matrix()
    fName = 'Data/att_lkf.txt'
    lkf = pd.read_csv(fName, sep=" ", header=None).as_matrix()
    fName = 'Data/att_ekf.txt'
    ekf = pd.read_csv(fName, sep=" ", header=None).as_matrix()
    return dcm[:-1,:], lkf[:-1,:], ekf[:-1,:]

if __name__ == '__main__':
    g, a, m = readAttRaw()
    print g.shape
    print a.shape
    print m.shape

    d, l, e = readAttEst()
    print d.shape
    print l.shape
    print e.shape
