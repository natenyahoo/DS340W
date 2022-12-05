# Notes: use batchnorm rather than layernorm 

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn


import time 




def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)



## time2vec

class Time2Vector(tf.keras.layers.Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                    shape=(int(self.seq_len),),
                                    initializer='uniform',
                                    trainable=True)

    def call(self, x):
        x = tf.math.reduce_mean(x[:,:,:], axis=-1) # Convert (batch, seq_len, 5) to (batch, seq_len)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1) # (batch, seq_len, 1)
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1) # (batch, seq_len, 2



## The Encoder system used inside of the Transformer model 

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


## The Transformer Model from https://www.kaggle.com/code/peressim/tps-jan-2022-transformer-with-time2vec/notebook

class Transformer(keras.Model):
    def __init__(
            self,
            num_hid=64, # embed_dim - num of features
            time_steps=7,
            num_head = 2,
            num_feed_forward=128, # pointwise dim
            num_layers_enc = 4,
            time_embedding = False,
    ):
        super().__init__()
        self.num_hid = num_hid
        if time_embedding:
            self.num_hid += 2
            self.tv = Time2Vector(time_steps)
        else:
            self.tv = None
        self.numlayers_enc = num_layers_enc
        self.enc_input = layers.Input((time_steps, self.num_hid))
        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(self.num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )
        self.GlobalAveragePooling1D = layers.GlobalAveragePooling1D(data_format='channels_last')
        self.out = layers.Dense(units=5, activation='linear')        
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        
    def call(self, inputs):
        if self.tv:
            x = self.tv(inputs)
            x = self.concat([inputs, x])
            x = self.encoder(x)
        else:
            x = self.encoder(inputs)
        x = self.GlobalAveragePooling1D(x)
        y = self.out(x)
        return y


def smape(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    num = tf.math.abs(tf.math.subtract(y_true, y_pred))
    denom = tf.math.add(tf.math.abs(y_true), tf.math.abs(y_pred))
    denom = tf.math.divide(denom,200.0)
    
    val = tf.math.divide(num,denom)
    val = tf.where(denom == 0.0, 0.0, val) 
    return tf.reduce_mean(val)


def rmse(predictions, targets):
    y_true = tf.cast(targets, tf.float32)
    y_pred = tf.cast(predictions, tf.float32)

    MSE = tf.math.square(tf.math.subtract(y_pred, y_true).mean())


    return tf.math.sqrt(MSE)
  
def mae(y_pred, y_true):
    return sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')


def fit_transformer_model(model, X, y, callbacks, epochs, batch_size, model_results_path):
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    opt = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError() # tf.keras.metrics.RootMeanSquaredError() # smape, rmse, mae 

    model.compile(optimizer=opt, loss=loss) #smape
    model.fit(x_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                callbacks=callbacks, 
                validation_data = (x_test, y_test),
                verbose=2)
    
    model.save(model_results_path) 

    history = [model.history]

    scores = [model.evaluate(x_test, y_test, verbose = 2)]

    return history, scores



def fit_transformer_model_w_K_fold(model, X, y, epochs, batch_size, callbacks, model_results_path, K = 3, **kwargs):
    
    scores = []
    histories = []
    for train_fold, test_fold in KFold(n_splits=K, shuffle=True).split(X,y):
        opt = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.MeanSquaredError() # tf.keras.metrics.RootMeanSquaredError() # smape, rmse, mae 

        model.compile(optimizer=opt, loss=loss) #smape
        histories.append(model.fit(X[train_fold], y[train_fold], epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                                   validation_data = (X[test_fold], y[test_fold]), # feed in the test data for plotting
                                   **kwargs).history)
        model.save(model_results_path)    
        scores.append(model.evaluate(X[test_fold], y[test_fold], verbose = 2)) # evaluate the test dataset
    print(scores)
    print(np.array(scores))
    print("average test loss: ", np.asarray(scores)[:,0].mean())
    print("average test accuracy: ", np.asarray(scores)[:,1].mean())
    print(model.summary())
    return scores, histories




############################################################


import config as fig

from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorboard.plugins.hparams import api as hp
import os
from datetime import datetime, date
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
import time


def cross_validate_transformer(X, y, model_results_path, K = 3, **kwargs):
    scores = []
    histories = []
    callbacks, model_path = callback_selector(early_stopping = fig.early_stopping, monitor = fig.monitor, patience = fig.patience, min_delta = fig.min_delta, learning_rate_scheduler_bool = fig.learning_rate_scheduler_bool, custom_verbose= fig.custom_verbose, metrics = fig.metrics, model_and_weights_saved = fig.model_and_weights_saved, tensorboard = fig.tensorboard, reduce_lr_on_plateau = fig.reduce_lr_on_plateau, model_results_path= fig.model_results_path)
    for train, test in KFold(n_splits=K, shuffle=True).split(X,y):
        model = Transformer() # compile model
        start = time.time()
        histories.append(model.fit(X[train], y[train], epochs = 10, batch_size = 32, callbacks=callbacks,
                                   validation_data = (X[test],y[test]), # feed in the test data for plotting
                                   **kwargs).history)
        print(time.time() - start)
        scores.append(model.evaluate(X[test], y[test], verbose = 2)) # evaluate the test dataset
    
    print(scores)
    print(np.array(scores))
    print("average test loss: ", np.asarray(scores)[:,0].mean())
    print("average test accuracy: ", np.asarray(scores)[:,1].mean())
    print(model.summary())
    return scores, histories


def plot_histories(histories, model_results_path, metrics = ['loss', 'accuracy', 'val_accuracy','val_loss']):
    """
    function to plot the histories of data
    """

    print("START OF PLOT HISTORIES")

    fig, axes = plt.subplots(nrows = (len(metrics) - 1) // 2 + 1, ncols = 2, figsize = (16,16))
    axes = axes.reshape((len(metrics) - 1) // 2 + 1, 2)
    for i,metric in enumerate(metrics):
        for history in histories:
            axes[(i+2)//2 - 1, 1 - (i+1)%2].plot(history[metric])
            axes[(i+2)//2 - 1, 1 - (i+1)%2].legend([i for i in range(len(histories))])
            axes[(i+2)//2 - 1, 1 - (i+1)%2].set_xticks(np.arange(max(history[metric])))
    
        plt.savefig(model_results_path + 'K-Fold #{}.png'.format(i))

    print("END OF PLOT HISTORIES")


########################################################
import time

from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorboard.plugins.hparams import api as hp
import os
from datetime import datetime, date
import math


class Custom_Verbose(Callback): #####################
    def __init__(self, metrics):
        self.metrics = metrics  
    
    def on_train_begin(self, logs = None): #####################
        print("The training session has begun...")
    
    def on_epoch_begin(self, epoch, logs = None):
        print("Epoch #{} has begun...".format(epoch+1)) #####################
    
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch #{} has ended.".format(epoch+1))
        print("\tResults for Epoch#{}:".format(epoch+1)) #####################
        #for i in self.metrics: 
        print("\t\t loss = {:7.2f}".format(logs["loss"]))
        #print("\t\t accuracy = {:7.2f}".format(logs['accuracy']))

    def on_train_end(self, logs = {}): #####################
        print("Training has ended.")
        print("\tTraining Results:")
        #for i in self.metrics: 
        print("\t\t loss = {:7.2f}".format(logs["loss"]))
        #print("\t\t accuracy = {:7.2f}".format(logs['accuracy']))


# def learning_rate_scheduler(epoch):
#     initial_lrate = 0.3
#     drop = 0.05
#     epochs_drop = 2.0
#     lrate = initial_lrate * math.pow(drop,  
#             math.floor((1+epoch)/epochs_drop))
#     return lrate

def learning_rate_scheduler(epoch):
    initial_rate = 0.3
    drop_rate = 0.035
    epoch_interval_to_drop = 2 # every "x" epochs you drop the initial rate by the drop rate

    lrate = initial_rate - ((epoch-1) * drop_rate)
    return lrate


def callback_selector(early_stopping, monitor, patience, min_delta, learning_rate_scheduler_bool, custom_verbose, metrics, model_and_weights_saved, tensorboard, reduce_lr_on_plateau, model_results_path):
    
    callbacks = [] 

    now = datetime.now()

    date_and_time = now.strftime("%d-%m-%Y-%H-%M-%S")

    #### ModelCheckpoint() / models_and_weights_saved

    saved_model_path = os.path.join(model_results_path,'Saved_Model_Results:{}'.format(date_and_time))
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)


    if model_and_weights_saved == 'both': 
        checkpoint = ModelCheckpoint(saved_model_path) #####################
        callbacks.append(checkpoint)

    elif model_and_weights_saved == 'model':
        checkpoint = ModelCheckpoint(saved_model_path, save_model_only = True) #####################
        callbacks.append(checkpoint) 

    elif model_and_weights_saved == 'weights':
        checkpoint = ModelCheckpoint(saved_model_path, save_weights_only = True)
        callbacks.append(checkpoint) 

    #### EarlyStopping / early_stopping 
    if early_stopping == True:
        early_stopping_object = EarlyStopping(monitor = monitor, patience = patience, min_delta = min_delta) ##################### finish 
        callbacks.append(early_stopping_object)

    #### LearningRateScheduler / learning_rate_scheduler

    if learning_rate_scheduler_bool == True: 
        learning_rate_scheduler_object = LearningRateScheduler(learning_rate_scheduler) ##################### finish 
        callbacks.append(learning_rate_scheduler_object)

    #### Custom_Verbose / custom_verbose
    if custom_verbose == True:
        callbacks.append(Custom_Verbose(metrics = metrics)) #####################

    model_path = ''
    if tensorboard == True: 
        
        tensorboard_model_path = os.path.join(saved_model_path, "Tensorboard_Results")
        if not os.path.exists(tensorboard_model_path):
            os.mkdir(tensorboard_model_path)

        callbacks.append(TensorBoard(log_dir = tensorboard_model_path, histogram_freq=1))
    
    if reduce_lr_on_plateau == True:
        callbacks.append(ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, min_lr= 0.001))


    #WHEN WE WANT TO DO HYPERPARAMETER TUNING: 
        # HP_NUM_UNITS=hp.HParam('num_units', hp.Discrete([64, 128]))
        # HP_DROPOUT=hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
        # HP_LEARNING_RATE= hp.HParam('learning_rate', hp.Discrete([0.001, 0.0005, 0.0001]))
        # HP_OPTIMIZER=hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
        # METRIC_ACCURACY='accuracy'
        
        # log_dir = '/home/openvessel/Documents/OpenVessel/Liver_Detection_2.0_Results/Tensorboard_Results/logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M')
        # with tf.summary.create_file_writer(log_dir).as_default():
        #     hp.hparams_config(
        #         hparams= [HP_NUM_UNITS, HP_DROPOUT,  HP_OPTIMIZER, HP_LEARNING_RATE],
        #         metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        #     )



##################### get rid of this? 
    # callback_object_list = [early_stopping, learning_rate_scheduler]

    # for i in callback_object_list: 
    #     if i == None:
    #         callback_object_list.pop(i) ##################### correct term "pop"? 
    #     else: 
    #         callbacks.append(i)

    # if early_stopping != False: 
    #     callbacks.append(early_stopping)
    
    # if learning_rate_scheduler != False: 
    #     callbacks.append(learning_rate_scheduler) #####################
    
    return callbacks, model_path
