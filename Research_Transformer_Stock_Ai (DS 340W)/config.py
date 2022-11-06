# What Input Options and Parameters do I want to be able to pass the model? 

## Input Options 
data_needed = True 
days_predicted = 3 
days_as_input = 20 

# technical_indicator Choices: []
technical_indicators = []

# fundamental_indicator Choices: []
fundamental_indicators = []

# economic_indicator Choices: []
economic_indicators = []

features = economic_indicators + fundamental_indicators + technical_indicators  

## Model Settings 


## Hyperparameter Settings 

# tuning_style Choices: [Particle Swarm, Population, Grid Search, ... ] 
tuning_bool = True 
tuning_style = "Particle Swarm" 

epochs = 10
learning_rate = 0.01
batch_size = 32 
# optimizer Choices: [adam, ...]
optimizer = "adam" 

parameters_to_tune = [learning_rate, epochs, batch_size]



## Training Session Settings 
save_model = True
stop_at_overfit = True 



# If SuperComputer 
supercomputer_bool = False

# If my GPU 


if supercomputer_bool: 
    # Connect to Supercomputer 

# else needed to access GPU? 

## Run It 

