#encoding:UTF-8

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from convnets import convnet
import datetime


def predict(result, thres=0.99):
    result[result>thres] = 1
    result[result<=thres] = 0
    result = np.argmax(result, axis=-1)
    neg = len(result[result==1])
    pos = len(result[result==2])
    if pos == neg == 0:
        return 0
    elif pos > neg:
        return 2
    else:
        return 1
        
def main():
    #-----------------------------------------------------------------
    # 1: Set some necessary parameters
    weights_path = 'model/convnet_227_weights_epoch08_loss0.0016.h5'

    #-----------------------------------------------------------------
    # 2: Build the Keras model
    model = convnet('alexnet', weights_path=weights_path, heatmap=True)
    
    posedge_path = '/home/zhaoke/justrypython/ks_idcard_ocr/testimg/card_bat/'
    negedge_path = '/home/zhaoke/justrypython/ks_idcard_ocr/testimg/neg_imgs/'
    background_path = '/home/zhaoke/gtest/ADEChallengeData2016/images/training2w/'
    
    starttime = datetime.datetime.now()
    print 'start time is ', starttime
    
    pos_cnt = 0
    pos_rgt = 0
    for i in os.listdir(posedge_path):
        img = cv2.imread(posedge_path+i)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        result = predict(result)
        pos_cnt += 1
        if result == 2:
            pos_rgt += 1
    
    neg_cnt = 0
    neg_rgt = 0
    for i in os.listdir(negedge_path):
        img = cv2.imread(negedge_path+i)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        result = predict(result)
        neg_cnt += 1
        if result == 1:
            neg_rgt += 1
    
    bck_cnt = 0
    bck_rgt = 0
    for i in os.listdir(background_path):
        img = cv2.imread(background_path+i)
        if img.shape[0] < 227 or img.shape[1] < 227:
            continue
        result = model.predict(np.array([img]))
        result = predict(result)
        bck_cnt += 1
        if result == 0:
            bck_rgt += 1
        if bck_cnt > 500:
            break
        
    print 'the posedge rate is %.3f'%(float(pos_rgt)/pos_cnt)
    print 'the negedge rate is %.3f'%(float(neg_rgt)/neg_cnt)
    print 'the background rate is %.3f'%(float(bck_rgt)/bck_cnt)
    print 'the total rate is %.3f'%(float(pos_rgt+neg_rgt+bck_rgt)/(pos_cnt+neg_cnt+bck_cnt))

    endtime = datetime.datetime.now()
    print 'the total time is ', endtime - starttime
    
    print 'end'
    
    

if __name__ == '__main__':
    main()