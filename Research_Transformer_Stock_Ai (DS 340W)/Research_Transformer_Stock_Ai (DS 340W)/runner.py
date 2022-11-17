
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from stock_transformer import *

# from trainer import *

import config as fig

# ######## Get Data ###########

# ## Using LSTM Sample Data 

# og_X_train_3d_shape = 28

# x_train = np.loadtxt("/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Dataset_X_train_Model_Ready.txt")

# x_train = x_train.reshape(
#     x_train.shape[0], x_train.shape[1] // og_X_train_3d_shape, og_X_train_3d_shape)

# y_train = pd.read_csv("/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Dataset_y_train_Model_Ready.csv")

# y_train = y_train.to_numpy()

# # Using the Raw Stock Dataset 

# # raw_stock_data = pd.read_csv("Sample_Stock_Data_Raw_No_IDs.csv")
# # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed, shuffle=False)
# # x_train = np.append(x_train, y_train.values.reshape(-1, 1), axis=1)
# # x_test = np.append(x_test, y_test.values.reshape(-1, 1), axis=1)
# # x_train, y_train = split_sequences(x_train, TIMESTEPS)
# # x_test, y_test = split_sequences(x_test, TIMESTEPS)


# ## Train Model 

# # Transformer Model Config 


# seed = 47
# TIMESTEPS = x_train.shape[1]

# num_heads=2
# num_layers_enc=2
# num_feed_forward=64
# num_features = x_train.shape[-1]
# time_steps = TIMESTEPS
# epochs = 20
# batch_size = 64


# ######## Training Config #########

# model_results_path = '/home/openvessel/Documents/Data Science/LSTM_Stock_Project/LSTM_Model_Results_2'
# weights_file_path = ""

# early_stopping = True 
# monitor = 'loss' 
# patience = 10
# min_delta = 0.005

# ## CustomVerbose() ... a custom callback 
# custom_verbose= False 
# train_verbose = True #if custom_verbose = False then would you like default verbose (True) or all verbose off (False) when training 

# ## learning_rate_scheduler
# learning_rate_scheduler_bool = True

# metrics = ["accuracy"]

# ## model_checkpoint callback creator. Choices: 'both', 'model', 'weights'
# model_and_weights_saved = 'both' 
# model_results_path = '/home/openvessel/Documents/Data Science/LSTM_Stock_Project/LSTM_Model_Results_2'

# ## tensorboard 
# tensorboard = True 

# ## ReduceLROnPlateau ? 
# reduce_lr_on_plateau = True 





######### Create and Fit Model  #################

#Data 

# if fig.use_LSTM_Sample_Data == True:_

x_train = np.loadtxt(fig.x_train_path)

x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1] // fig.og_X_train_3d_shape, fig.og_X_train_3d_shape)

y_train = pd.read_csv(fig.y_train_path)

y_train = y_train.to_numpy()



model = Transformer(num_hid=fig.num_features,
                        time_steps=fig.time_steps,
                        time_embedding=True,
                        num_head=fig.num_heads,
                        num_layers_enc=fig.num_layers_enc,
                        num_feed_forward=fig.num_feed_forward)

fit_transformer_model(model, x_train, y_train, fig.epochs, fig.batch_size)


## Evaluate Results 


# Transformer's way to evaluate performance 

# results = model.evaluate(x_test, y_test)
# print(results)


# Using my LSTM Code to Evaluate 

scores, histories = cross_validate_transformer(x_train, y_train, fig.model_results_path)
plot_histories(histories)