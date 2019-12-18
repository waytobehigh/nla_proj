# -*- coding: utf-8 -*-
""" Script for running RNNs with fixed parameters. """

import os
import sys
import time
import math
import numpy as np
import csv
import Params
import load
import rnn
import matplotlib.pyplot as plt

def train(params):

    print('%s starting......' % params.cell)
    sys.stdout.flush()

    if params.dataset.startswith('mnist'):
        train_X, test_X, train_y, test_y = load.load_mnist(params)
    elif params.dataset.startswith('add'):
        train_X, test_X, train_y, test_y = load.adding_task(params)
    else:
        assert 0, "unknown dataset %s" % (params.dataset)

    print ("parameters = ", params)

    model = rnn.RNNModel(params)

    # load model
    if params.load_model:
        model.load("%s" % (params.load_model_dir))

    # train model
    train_error, test_error,epochs = model.train(params, train_X, train_y, test_X, test_y)
    
    #save data to file(Egor)
    
    with open('data_'+params.cell+'_dataset_'+params.dataset+'_L_'+str(params.num_layers)+'_rsize_'+str(params.r_size),'w') as file:
        for i in range(len(train_error)):
            file.write(str(epochs[i])+' '+str(train_error[i])+' '+str(test_error[i])+'\n')
    
    
    # save model
    
    if params.model_dir:
        if os.path.isdir(os.path.dirname(params.model_dir)) == False:
            os.makedirs(params.model_dir)
        model.save("%s.%s" % (params.model_dir, params.cell))

    # predict
    train_pred = model.predict(train_X, params.batch_size)
    test_pred = model.predict(test_X, params.batch_size)

    # must close model when finish
    model.close()

    # write prediction to file
    if params.pred_dir:
        if os.path.isdir(os.path.dirname(params.pred_dir)) == False:
            os.makedirs(params.pred_dir)
        with open("%s.%s.%s.y" % (params.pred_dir, params.dataset, params.cell), "w") as f:
            content = ""
            for pred in [train_pred, test_pred]:
                for entry in pred:
                    for index, value in enumerate(entry):
                        if index:
                            content += ","
                        content += "%f" % (value)
                    content += "\n"
            f.write(content)
        with open("%s.%s.%s.X" % (params.pred_dir, params.dataset, params.cell), "w") as f:
            content = ""
            for X in [train_X, test_X]:
                for entry in X:
                    for index, value in enumerate(entry.ravel()):
                        if index:
                            content += ","
                        content += "%f" % (value)
                    content += "\n"
            f.write(content)
    return train_error, test_error

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("input parameters in json format in required")
        exit()
    paramsArray = []
    for i in range(1, len(sys.argv)):
        params = Params.Params()
        params.load(sys.argv[i])
        paramsArray.append(params)
    print("parameters[%d] = %s" % (len(paramsArray), paramsArray))

    tt = time.time()
    for params in paramsArray:
        train(params)
    print("program takes %.3f seconds" % (time.time()-tt))