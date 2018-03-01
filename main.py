from PrepData import *
import sys
import numpy as np
import matplotlib.pyplot as plt
from Model import *

def getSeqInput(x,y, T):

    y_prev = np.concatenate([np.zeros((1,3)), y], axis=0)[:-1,:]
    x = np.concatenate([y_prev, x], axis=1)
    input = None
    offset = T-1
    zeropads = np.zeros((offset, x.shape[1]))
    x = np.concatenate([zeropads, x], axis=0)
    for i in range(0, x.shape[0]-offset):
        temp = np.reshape(x[i:i+T, :], (1, -1,12))
        input = temp if input is None else np.concatenate([input, temp], axis=0)
    label = y[0:y.shape[0],:]
    #label = np.concatenate([np.zeros((label.shape[0],3)), label], axis=1)
    return input, label

def train():
    gyro, acc, mag = readAttRaw()
    dcm, lkf, ekf= readAttEst()

    x = np.concatenate([gyro, acc, mag], axis = 1)
    y = lkf*180/np.pi

    T = 10
    input, label = getSeqInput(x,y,T)
    model = getEuler(input.shape[1:], T)
    #model.load_weights('temp.h5')
    history = model.fit(input, label, epochs=2000, batch_size=1024, verbose=1, shuffle=True)
    loss_history = history.history["loss"]
    model.save_weights('temp.h5')

def test():
    gyro, acc, mag = readAttRaw()
    dcm, lkf, ekf= readAttEst()
    x = np.concatenate([gyro, acc, mag], axis = 1)
    y = lkf*180/np.pi
    T = 10
    input, label = getSeqInput(x,y,T)

    model = getEuler(input.shape[1:], T)
    model.load_weights('temp.h5')
    p = model.predict(input)

    plt.figure()
    gt_qkf = plt.plot(y, 'go')
    gt_dcm = plt.plot(dcm*180/np.pi, 'r')
    pr = plt.plot(p, 'b')
    plt.legend([gt_dcm[0], gt_qkf[0], pr[0]], ['DCM', 'Q-KF', 'LSTM'])
    plt.show()





if __name__ == '__main__':
    type = int(sys.argv[1])
    if type==0:
        train()
    elif type==1:
        test()
