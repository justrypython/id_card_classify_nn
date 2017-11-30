#encoding:UTF-8

import os
import numpy as np
import tensorflow as tf
import h5py
from keras.optimizers import SGD
from convnets import convnet

def main():
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet', heatmap=False)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    
    with h5py.File('model/alexnet_weights.h5', 'r') as f:
        names = f.keys()
        for i in names:
            print i
            for j in f[i].keys():
                print f[i][j].shape
        print 'print h5 end'
    layers = model.layers
    for i in layers:
        print i.name
        for j in i.get_weights():
            print j.shape
    print 'print model end'
    for i in range(5):
        print '\n'
    f = h5py.File('model/alexnet_weights.h5', 'r')
    for i in layers:
        name = i.name
        if 'conv_' in name:
            if len(i.get_weights()) > 0:
                print 'load %s weights'%name
                w = f[name][name+'_W']
                b = f[name][name+'_b']
                w = np.transpose(w, [2, 3, 1, 0])
                i.set_weights([w, b])
            else:
                print 'layers %s has no weights'%name
        elif 'dense_' in name and name != 'dense_4':
            print 'load %s weights'%name
            w = f[name][name+'_W']
            b = f[name][name+'_b']
            i.set_weights([w, b])
    print 'load end!!!'
    print 'save weights'
    model.save_weights('model/tf_alexnet_weights.h5')
        
        
if __name__ == '__main__':
    main()