### Taking Data and Making it ready for the Stock Transformer 

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


## be able to handle multiple time lengths 

# ## standardize and normalize 
# standardized_stock_features = StandardScaler().fit_transform(relevant_stock_features)

# ## PCA 
# pca = PCA(0.9)

# pca.fit(standardized_stock_features)

# print("# of Components needed for 90% Variance explained =", pca.n_components_) 

# pca_stock_features = pca.transform(standardized_stock_features)

# pca_stock_features_df = pd.DataFrame(pca_stock_features)



## Time-Series Transformation: 
    # 2-D Dataset (days by features) --> Time-Series (3-D) (time-series, many days, features) 


## EXAMPLE: (30 days as input for 5 days prediction)

# X_train = pd.DataFrame()
# y_train = pd.DataFrame()

# curr_X_train = []
# curr_y_train = []



# #   for row_num in range(0,len(data))
# for row_num in range(0, len(fully_standardized_data_reordered)-36):

#   # if row_num's symbol == row_num + 34 's symbol :
#   if fully_standardized_data_reordered["symbol"].iloc[row_num] == fully_standardized_data_reordered["symbol"].iloc[row_num+34]:
#     #   input array = [row_num : row_num + 31) of data that is not the first 2 rows
#     #   output label array = [31:36) of data that is not the first 2 rows
#     input_matrix = fully_standardized_data_reordered.iloc[row_num : row_num+30, 2:]
#     output_matrix = fully_standardized_data_reordered.iloc[row_num+30:row_num+35,3]
#     curr_X_train.append(input_matrix)
#     curr_y_train.append(output_matrix)
#   #   append input array to X_train 
#   #   append out_put label array to y_train 

# X_train = np.array(curr_X_train)
# y_train = np.array(curr_y_train)

# print(X_train.shape)
# print(y_train.shape)

result = pd.read_csv("/home/openvessel/Documents/Data Science/DS_320_Stock_RNN/Fully_Standardized_Dataset.csv")

print(result.head)