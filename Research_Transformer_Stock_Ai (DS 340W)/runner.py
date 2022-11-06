import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from trainer import cross_validate, plot_histories
from stock_transformer import split_sequences

## Get Data 

unshaped_X_train = np.loadtxt("Sample_Stock_Dataset_X_train.txt")

my_X_train = unshaped_X_train.reshape(
    unshaped_X_train.shape[0], unshaped_X_train.shape[1] // unshaped_X_train.shape[2], unshaped_X_train.shape[2])

my_y_train = pd.read_csv("Sample_Stock_Dataset_y_train.csv")


## Train Model 

seed = 47
TIMESTEPS = 1 

# load in dataset 

raw_stock_data = pd.read_csv("Sample_Stock_Data_Raw_No_IDs.csv")

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed, shuffle=False)

x_train = np.append(x_train, y_train.values.reshape(-1, 1), axis=1)
x_test = np.append(x_test, y_test.values.reshape(-1, 1), axis=1)
x_train, y_train = split_sequences(x_train, TIMESTEPS)
x_test, y_test = split_sequences(x_test, TIMESTEPS)


num_heads=2
num_layers_enc=2
num_feed_forward=64
num_features = x_train.shape[-1]
time_steps = TIMESTEPS
epochs = 100
batch_size = 128

model = Transformer(num_hid=num_features,
                        time_steps=time_steps,
                        time_embedding=True,
                        num_head=num_heads,
                        num_layers_enc=num_layers_enc,
                        num_feed_forward=num_feed_forward)

opt = tf.keras.optimizers.Adam()
loss = tf.keras.losses.mse
model.compile(optimizer=opt, loss=smape)
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
print()
results = model.evaluate(x_test, y_test)
print(results)

## Evaluate Results 

scores, histories = cross_validate(X_train, y_train)
plot_histories(histories)