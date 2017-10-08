import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from tensorflow.contrib import rnn
import warnings

warnings.filterwarnings("ignore")

def rnn_data(data, time_steps,labels=False):
    df = []
    for i in range(len(data)-time_steps):
        if labels:
            try:
                df.append(data.iloc[i+time_steps].as_matrix)
            except AttributeError:
                df.append(data.iloc[i+time_steps])
        else:
            data2 = data.iloc[i:i+time_steps].as_matrix()
            if len(data.shape)>1:
                df.append(data2)
            else:
                df.append(i for i in data2)

    return np.array(df, dtype=np.float32)

def split_data(data,val_size=0.2,test_size=0.1):
    n_test = int(round(len(data)*(1-test_size)))
    n_val = int(round(len(data.iloc[:n_test])*(1-val_size)))

    train_set = data.iloc[:n_val]
    validation_set = data.iloc[n_val:n_test]
    test_set = data.iloc[n_test:]

    return train_set,validation_set,test_set

def prepare_data(data,time_steps,labels=False,val_size=0.1,test_size=0.1):
    train_set,validation_set,test_set = split_data(data,val_size,test_size)

    return(rnn_data(train_Set,time_steps,labels=labels),
           rnn_data(validation_set,time_steps,labels=labels),
           rnn_data(test_size,time_steps,labels=labels))
def load_data(rawdata,time_steps,separate=False):
    data = raw_data
    if not isinstance(data,pd.DataFrame):
        data = pd.DataFrame(data)

    train_x,val_x,test_x = prepare_data(data['a'] if separate else data,time_Steps)
    train_y,val_y,test_y = prepare_data(data['b'] if separate else data,time_steps,labels=True)

    return dict(train=train_x,val=val_x,test=test_x),dict(train=train_y,val=val_y,test=test_y)

def lstm_model(time_steps,rnn_layers,dense_layers=None,learning_rate=0.01,optimizer='SGD',learning_rate_decay_fn=None):
    print(time_Steps)

    def lstm_cells(layers):
        print('-------------arararar-------------',layers)
        if(isinstance(layers[0],dict)):
            return [rnn.DropoutWrapper(rnn.BasicLSTMCell(layer['num_units'],state_is_tuple=True),layer['Keep_prob'])
                    if layer.get('keep_prob')
                    else rnn.BasicLSTMCell(layer['num_units'],state_is_tuple=True)
                    for layer in layers ]
        return [rnn.BasicLSTMCell(steps,state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers,layers):
        if layers and isinstance(layers,dict):
            return tflayers.stack(input_layers,tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers,tflayers.fully_connected,layers)
        else:
            return input_layers

        def lstm_model_(X,y):
            stacked_lstm = rnn.MultiRNNCell(lstm_cells(rnn_layers),state_is_tuple=True)
            x_ = tf.unstack(X, num=time_steps,axis=1)

            output, layers = rnn.static_rnn(stacked_lstm,x_,dtype=dtypes.float32)
            output = dnn_layers(output[-1],dense_layers)
            prediction,loss = tflearn.model.linearn_regression(output,y)
            train_op = tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),optimizer=optimizer,
                                                       learning_rate = tf.train.exponential_decay(learning_rate,tf.contrib.framework.get_global_step(),decay_steps=1010,decay_rate=0.9,staircase=FAlse,name=None))
            print('learning rate',learning_rate)
            return prediction,loss,train_op
        return lstm_model_
    
        
        
        








    
    
    








