import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from sklearn.metrics import mean_squared_error as MSE

from . import lstmTimeSeriesStacked as l 
warnings.filterwarnings("ignore")

LOG_DIR = 'resources/logs/'
TIMESTEPS = 1
RNN_LAYERS = [{'num_units':500}]
DENSE_LAYERS = 4
TRAINING_STEPS = 500
PRINT_STEPS = TRAINING_STEPS/10
BATCH_SIZE = 100

regressor = SKCompat(learn.Estimator(model_fn=l.lstm_model(TIMESTEPS,RNN_LAYERS,DENSE_LAYERS),model_dir=LOG_DIR))
X,y = l.Sgenerate_data(np.sin,np.linspace(0,100,10000,dtype=np.float32),TIMESTEPS,seperate=False)
noise_train = np.asmatrix(np.random.normal(0,0.2,len(y['train'])),dtype = np.float32)
noise_val = np.asmatrix(np.random.normal(0,0.2,len(y['val'])),dtype = np.float32)
noise_test = np.asmatrix(np.random.normal(0,0.2,len(y['test'])),dtype = np.float32)

y['train'] = np.add(y['train'],noise_train)
y['test'] = np.add(y['test'],noise_test)
y['val'] = np.add(y['val'],noise_val)

print('---------------------------------')
print('train y shape',y['train'].shape)
print('train y shape_num', y['train'][1:5])
print('noisw_train shape',noise_train.shape)
print('noise_train shape_num',noise_train.shape[1:5])

validation_monitor = learn.monitors.ValidationMonitors(X['val'],y['val'],every_n_steps=PRINT_STEPS)

SKCompat(regressor.fit(X['train'],y['train'],
                       monitors=[validation_monitor],
                       batch_size=BATCH_SIZE,
                       steps=TRAINING_STEPS))
print('X train shape', X['train'].shape)
print('y_train shape', y['train'].shape)

print('X test shape', X['test'].shape)
print('y test shape', y['test'].shape)

predicted = np.asmatrix(regressor.predict(X['test']),dtype=np.float32)
predicted = np.transpose(predicted)

RMSE = np.sqrt((np.asarray((np.subtract(predicted,y['test'])))**2).mean())

score = MSE(predicted,y['test'])
NMSE = score/np.var(y['test'])

print("RMSE: %f" % RMSE)
print("NMSE: %f" % NMSE)
print("MSE: %f" % score)

plot_test = plt.plot(y['test'],label='test')
plot_predicted = plt.plot(predicted,label='predicted')
plt.legend(handles=[plot_predicted,plot_test])
plt.show()


























