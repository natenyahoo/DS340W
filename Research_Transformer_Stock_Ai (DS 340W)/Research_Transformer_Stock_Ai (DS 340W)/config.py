# What Input Options and Parameters do I want to be able to pass the model? 

# ## Input Options 
# data_needed = True 
# days_predicted = 3 
# days_as_input = 20 

# # technical_indicator Choices: []
# technical_indicators = []

# # fundamental_indicator Choices: []
# fundamental_indicators = []

# # economic_indicator Choices: []
# economic_indicators = []

# features = economic_indicators + fundamental_indicators + technical_indicators  

# ## Model Settings 
#        #### train_LD_model()
# validation_split = 0.20
# epochs = 10
# batch_size = 3

## Hyperparameter Settings 

## tuning_style Choices: [Particle Swarm, Population, Grid Search, ... ] 
# tuning_bool = True 
# tuning_style = "Particle Swarm" 

# epochs = 10
# learning_rate = 0.01
# batch_size = 32 
# # optimizer Choices: [adam, ...]
# optimizer = "adam" 

# parameters_to_tune = [learning_rate, epochs, batch_size]



## Training Session Settings 
# save_model = True
# stop_at_overfit = True 


#### Liver_Detection.py
        #### build_LD_model()
# dropout_rate = 0.2

# regularizer_choice = 'l1_l2' 
# l1_weight = 0.05 
# l2_weight = 0.001

#         #### load_weights_into_LD_model()
# load_weights = True 
# weights_file_path = '/home/openvessel/Documents/OpenVessel/Liver_Detection_2.0/liverseg-2017-nipsws/Liver_Detection_2.0/Liver_Detection-version:21/07/2021 02:07:30/variables/variables.data-00000-of-00001' 

#         #### load_model_into_LD_model()
# load_model = True 
# model_to_load = '/home/openvessel/Documents/OpenVessel/Liver_Detection_2.0/liverseg-2017-nipsws/Liver_Detection_2.0/Liver_Detection-version:21/07/2021 02:07:30'

#         #### compile_LD_model
# loss_function = "binary_crossentropy" 
# optimizer = "adam" 
# metrics = ["accuracy"]



#         #### Callback Selector 
#                 ## EarlyStopping
# early_stopping = True 
# monitor = 'accuracy' 
# patience = 3
# min_delta = 0.1

#                 ## CustomVerbose() ... a custom callback 
# custom_verbose= False 
# train_verbose = True #if custom_verbose = False then would you like default verbose (True) or all verbose off (False) when training 

#                 ## learning_rate_scheduler
# learning_rate_scheduler_bool = True

#                 ## model_checkpoint callback creator. Choices: 'both', 'model', 'weights'
# model_and_weights_saved = 'both' 
# model_results_path = '/home/openvessel/Documents/Data Science/LSTM_Stock_Project/LSTM_Model_Results'

#                 ## tensorboard 
# tensorboard = True 
#                 ## ReduceLROnPlateau ? 
# reduce_lr_on_plateau = True 

#         #### test_LD_model()
# test_verbose = 2



# # If SuperComputer 
# supercomputer_bool = False

# # If my GPU 


# if supercomputer_bool: 
#     # Connect to Supercomputer 

# # else needed to access GPU? 

# ## Run It 





######## Config from Config_and_Run (previous file name for runner.py )




######## Get Data ###########

## Using LSTM Sample Data 

use_LSTM_Sample_Data = True

og_X_train_3d_shape = 28

x_train_path = "/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Dataset_X_train_Model_Ready.txt"

y_train_path = "/home/openvessel/Documents/Data Science/LSTM_Stock_Project/Sample_Stock_Training_Data/Sample_Stock_Dataset_y_train_Model_Ready.csv"

# Using the Raw Stock Dataset 

# raw_stock_data = pd.read_csv("Sample_Stock_Data_Raw_No_IDs.csv")
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed, shuffle=False)
# x_train = np.append(x_train, y_train.values.reshape(-1, 1), axis=1)
# x_test = np.append(x_test, y_test.values.reshape(-1, 1), axis=1)
# x_train, y_train = split_sequences(x_train, TIMESTEPS)
# x_test, y_test = split_sequences(x_test, TIMESTEPS)


## Train Model 

# Transformer Model Config 


seed = 47
TIMESTEPS = 30 ## x_train.shape[1]

num_heads=2
num_layers_enc=2
num_feed_forward=64
num_features = 28 ## x_train.shape[-1]
time_steps = TIMESTEPS

epochs = 20
batch_size = 64 ## 128


######## Training Config #########



early_stopping = True 
monitor = 'loss' 
patience = 4
min_delta = 0.01

## CustomVerbose() ... a custom callback 
custom_verbose= False 
train_verbose = True #if custom_verbose = False then would you like default verbose (True) or all verbose off (False) when training 

## learning_rate_scheduler
learning_rate_scheduler_bool = True

metrics = ["accuracy"]

## model_checkpoint callback creator. Choices: 'both', 'model', 'weights'
model_and_weights_saved = 'both' 
model_results_path = '/home/openvessel/Documents/Data Science/Capstone_Stock_Ai/Research_Transformer_Stock_Ai (DS 340W)/Transformer_Model_Results'
weights_file_path = ""

## tensorboard 
tensorboard = True 

## ReduceLROnPlateau ? 
reduce_lr_on_plateau = True 
